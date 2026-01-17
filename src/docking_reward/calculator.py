"""Main RewardCalculator orchestrator."""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from rdkit import Chem

from .config import Config, load_config
from .docking import DockingResult, dock_to_targets
from .druglikeness import DruglikenessScorer
from .interactions import analyze_interactions, score_interactions
from .ligand_prep import (
    embed_molecule,
    mol_to_pdbqt,
    mol_to_sdf,
    sanitize_smiles_for_filename,
    smiles_to_mol,
)
from .parallel import Executor, get_executor
from .protein_prep import prepare_proteins

logger = logging.getLogger(__name__)

# Sentinel score for invalid/failed molecules
FAILURE_SCORE = -10.0


@dataclass
class ScoringContext:
    """
    Context object passed to worker processes containing all data needed for scoring.

    This is serialized and sent to worker processes, so all paths must be absolute
    and all objects must be picklable.
    """

    config: Config
    protein_pdbqts: dict[str, Path]
    temp_dir: Path
    output_dir: Optional[Path]
    save_poses: bool


@dataclass
class ScoringTask:
    """A single molecule scoring task."""

    index: int
    smiles: str
    context: ScoringContext


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of score components for a single molecule."""

    # Per-target docking scores (raw Vina scores, kcal/mol)
    docking_scores: dict[str, float]
    # Per-target docking contributions (after weight and negation)
    docking_contributions: dict[str, float]
    # Per-target interaction scores
    interaction_scores: dict[str, float]
    # Per-target best interaction pose index
    interaction_pose_idx: dict[str, int]
    # Drug-likeness scores
    qed_score: Optional[float] = None
    qed_contribution: Optional[float] = None
    custom_scores: dict[str, float] = None
    # Total
    total_score: float = 0.0

    def __post_init__(self):
        if self.custom_scores is None:
            self.custom_scores = {}


@dataclass
class ScoringResult:
    """Result of scoring a single molecule."""

    index: int
    smiles: str
    score: float
    pose_path: Optional[Path] = None
    error: Optional[str] = None
    breakdown: Optional[ScoreBreakdown] = None


def _score_single_molecule(task: ScoringTask) -> ScoringResult:
    """
    Score a single molecule (runs in worker process).

    This function is designed to be called by parallel executors.

    Args:
        task: ScoringTask containing all information needed

    Returns:
        ScoringResult with score and optional pose path
    """
    ctx = task.context
    smiles = task.smiles
    index = task.index

    # Parse SMILES
    mol = smiles_to_mol(smiles)
    if mol is None:
        return ScoringResult(
            index=index,
            smiles=smiles,
            score=FAILURE_SCORE,
            error="Invalid SMILES",
        )

    # Generate 3D conformer
    mol_3d = embed_molecule(mol)
    if mol_3d is None:
        return ScoringResult(
            index=index,
            smiles=smiles,
            score=FAILURE_SCORE,
            error="3D embedding failed",
        )

    # Prepare ligand PDBQT
    ligand_pdbqt = ctx.temp_dir / "ligands" / f"mol_{index}.pdbqt"
    if not mol_to_pdbqt(mol_3d, ligand_pdbqt):
        return ScoringResult(
            index=index,
            smiles=smiles,
            score=FAILURE_SCORE,
            error="PDBQT conversion failed",
        )

    # Dock to all targets
    docking_results = dock_to_targets(
        ligand_pdbqt=ligand_pdbqt,
        targets=ctx.config.targets,
        protein_pdbqts=ctx.protein_pdbqts,
        global_vina_config=ctx.config.vina,
    )

    # Check if any docking succeeded
    any_success = any(r.success for r in docking_results.values())
    if not any_success:
        errors = [f"{name}: {r.error}" for name, r in docking_results.items() if r.error]
        return ScoringResult(
            index=index,
            smiles=smiles,
            score=FAILURE_SCORE,
            error=f"Docking failed: {'; '.join(errors)}",
        )

    # Calculate total score and track breakdown
    total_score = 0.0

    # Initialize breakdown tracking
    breakdown = ScoreBreakdown(
        docking_scores={},
        docking_contributions={},
        interaction_scores={},
        interaction_pose_idx={},
    )

    # Collect all poses for SDF output
    all_poses: list[Chem.Mol] = []

    for target in ctx.config.targets:
        result = docking_results.get(target.name)
        if result is None or not result.success:
            breakdown.docking_scores[target.name] = None
            breakdown.docking_contributions[target.name] = 0.0
            breakdown.interaction_scores[target.name] = 0.0
            continue

        # Docking score contribution (negated so better binding = positive)
        docking_contribution = target.weight * (-result.best_score)
        total_score += docking_contribution

        # Track in breakdown
        breakdown.docking_scores[target.name] = result.best_score
        breakdown.docking_contributions[target.name] = docking_contribution

        logger.debug(
            f"Target '{target.name}': Vina={result.best_score:.2f}, "
            f"weight={target.weight}, contribution={docking_contribution:.2f}"
        )

        # Interaction scoring (evaluate poses within energy threshold of best)
        max_interaction_score = 0.0
        best_pose_idx = 0

        if target.interactions and result.poses and result.scores:
            best_energy = result.best_score  # Most negative = best
            threshold = target.interaction_energy_threshold

            # Find all poses within threshold of best
            # e.g., if best is -9.0 and threshold is 1.0, include poses with score <= -8.0
            for pose_idx, (pose, score) in enumerate(zip(result.poses, result.scores)):
                if pose is None:
                    continue
                # Check if this pose is within threshold (remember: more negative = better)
                if score <= best_energy + threshold:
                    interaction_counts = analyze_interactions(pose, target.pdb_path)
                    pose_interaction_score = score_interactions(
                        interaction_counts, target.interactions
                    )
                    if pose_interaction_score > max_interaction_score:
                        max_interaction_score = pose_interaction_score
                        best_pose_idx = pose_idx
                    logger.debug(
                        f"Pose {pose_idx} (E={score:.2f}) interaction score: {pose_interaction_score:.2f}"
                    )

            total_score += max_interaction_score
            logger.debug(
                f"Interaction score for '{target.name}': {max_interaction_score:.2f} "
                f"(from pose {best_pose_idx}, threshold={threshold} kcal/mol)"
            )

        # Track interaction scores in breakdown
        breakdown.interaction_scores[target.name] = max_interaction_score
        breakdown.interaction_pose_idx[target.name] = best_pose_idx

        # Collect poses with target info as properties
        for i, pose in enumerate(result.poses):
            if pose is not None:
                pose.SetProp("_Target", target.name)
                pose.SetProp("_Pose", str(i + 1))
                pose.SetProp("_VinaScore", f"{result.scores[i]:.2f}")
                all_poses.append(pose)

    # Drug-likeness contribution
    druglikeness_scorer = DruglikenessScorer(ctx.config.druglikeness)
    druglikeness_score = druglikeness_scorer.score(mol_3d)
    total_score += druglikeness_score
    logger.debug(f"Drug-likeness score: {druglikeness_score:.2f}")

    # Track drug-likeness in breakdown
    if ctx.config.druglikeness.qed:
        from .druglikeness import calculate_qed
        breakdown.qed_score = calculate_qed(mol_3d)
        breakdown.qed_contribution = ctx.config.druglikeness.qed.weight * breakdown.qed_score

    breakdown.total_score = total_score

    # Save poses if requested
    pose_path = None
    if ctx.save_poses and ctx.output_dir and all_poses:
        safe_smiles = sanitize_smiles_for_filename(smiles)
        pose_path = ctx.output_dir / "poses" / f"mol_{index}_{safe_smiles}.sdf"
        pose_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            writer = Chem.SDWriter(str(pose_path))
            for pose in all_poses:
                writer.write(pose)
            writer.close()
        except Exception as e:
            logger.warning(f"Failed to save poses for mol {index}: {e}")
            pose_path = None

    return ScoringResult(
        index=index,
        smiles=smiles,
        score=total_score,
        pose_path=pose_path,
        breakdown=breakdown,
    )


class RewardCalculator:
    """
    Main class for calculating docking-based reward scores.

    Orchestrates ligand preparation, docking, interaction analysis,
    and drug-likeness scoring with parallel execution.
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        n_workers: int = 1,
        backend: str = "multiprocessing",
        temp_dir: Optional[Path] = None,
    ):
        """
        Initialize the reward calculator.

        Args:
            config_path: Path to YAML configuration file
            n_workers: Number of parallel workers
            backend: "multiprocessing" or "dask"
            temp_dir: Directory for temporary files (default: creates tmp/ subdir in cwd)
        """
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self.n_workers = n_workers
        self.backend = backend

        # Set up temp directory
        if temp_dir is None:
            self.temp_dir = Path.cwd() / "tmp"
        else:
            self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Prepare proteins once at initialization
        logger.info("Preparing protein structures...")
        self.protein_pdbqts = prepare_proteins(self.config.targets, self.temp_dir)
        logger.info(f"Prepared {len(self.protein_pdbqts)} protein structures")

        # Executor is created lazily
        self._executor: Optional[Executor] = None

    def _get_executor(self) -> Executor:
        """Get or create the parallel executor."""
        if self._executor is None:
            self._executor = get_executor(self.backend, self.n_workers)
        return self._executor

    def score(self, smiles_list: list[str]) -> list[float]:
        """
        Score a list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings to score

        Returns:
            List of reward scores (same order as input)
            Invalid SMILES or failed docking returns -10.0
        """
        if not smiles_list:
            return []

        # Create scoring context
        context = ScoringContext(
            config=self.config,
            protein_pdbqts=self.protein_pdbqts,
            temp_dir=self.temp_dir,
            output_dir=None,
            save_poses=False,
        )

        # Create tasks
        tasks = [
            ScoringTask(index=i, smiles=s, context=context)
            for i, s in enumerate(smiles_list)
        ]

        # Execute in parallel
        executor = self._get_executor()
        results = executor.map(_score_single_molecule, tasks)

        # Extract scores in order
        scores = [FAILURE_SCORE] * len(smiles_list)
        for result in results:
            scores[result.index] = result.score
            if result.error:
                logger.debug(f"Molecule {result.index} ({result.smiles}): {result.error}")

        return scores

    def score_file(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> Path:
        """
        Score SMILES from a file and write results.

        Args:
            input_path: Path to file with one SMILES per line
            output_dir: Directory for output files

        Returns:
            Path to results.csv

        Creates:
            output_dir/
            ├── results.csv
            └── poses/
                ├── mol_0_<smiles>.sdf
                └── ...
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Read SMILES from file
        smiles_list = []
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    smiles_list.append(line)

        if not smiles_list:
            logger.warning(f"No SMILES found in {input_path}")
            # Write empty results
            results_path = output_dir / "results.csv"
            with open(results_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["smiles", "score"])
            return results_path

        logger.info(f"Scoring {len(smiles_list)} molecules...")

        # Create scoring context with output directory for poses
        context = ScoringContext(
            config=self.config,
            protein_pdbqts=self.protein_pdbqts,
            temp_dir=self.temp_dir,
            output_dir=output_dir,
            save_poses=True,
        )

        # Create tasks
        tasks = [
            ScoringTask(index=i, smiles=s, context=context)
            for i, s in enumerate(smiles_list)
        ]

        # Execute in parallel
        executor = self._get_executor()
        results = executor.map(_score_single_molecule, tasks)

        # Sort results by index to maintain order
        results_sorted = sorted(results, key=lambda r: r.index)

        # Write results CSV
        results_path = output_dir / "results.csv"
        with open(results_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["smiles", "score"])
            for result in results_sorted:
                writer.writerow([result.smiles, f"{result.score:.4f}"])

        # Write breakdown CSV
        breakdown_path = output_dir / "breakdown.csv"
        self._write_breakdown_csv(breakdown_path, results_sorted)

        # Log summary
        scores = [r.score for r in results_sorted]
        valid_scores = [s for s in scores if s != FAILURE_SCORE]
        n_valid = len(valid_scores)
        n_failed = len(scores) - n_valid

        logger.info(
            f"Scoring complete: {n_valid} succeeded, {n_failed} failed"
        )
        if valid_scores:
            logger.info(
                f"Score range: {min(valid_scores):.2f} to {max(valid_scores):.2f}, "
                f"mean: {sum(valid_scores) / len(valid_scores):.2f}"
            )

        return results_path

    def _write_breakdown_csv(self, path: Path, results: list[ScoringResult]) -> None:
        """
        Write detailed score breakdown CSV.

        Args:
            path: Output path for breakdown.csv
            results: List of ScoringResult objects with breakdowns
        """
        # Build header dynamically based on targets
        target_names = [t.name for t in self.config.targets]

        header = ["smiles", "total_score", "error"]

        # Add columns for each target
        for name in target_names:
            header.extend([
                f"{name}_vina_score",
                f"{name}_docking_contribution",
                f"{name}_interaction_score",
                f"{name}_interaction_pose",
            ])

        # Add drug-likeness columns
        header.extend(["qed_score", "qed_contribution"])

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for result in results:
                row = [
                    result.smiles,
                    f"{result.score:.4f}",
                    result.error or "",
                ]

                if result.breakdown:
                    bd = result.breakdown
                    for name in target_names:
                        vina = bd.docking_scores.get(name)
                        row.append(f"{vina:.2f}" if vina is not None else "")
                        row.append(f"{bd.docking_contributions.get(name, 0):.4f}")
                        row.append(f"{bd.interaction_scores.get(name, 0):.4f}")
                        row.append(str(bd.interaction_pose_idx.get(name, "")))

                    row.append(f"{bd.qed_score:.4f}" if bd.qed_score is not None else "")
                    row.append(f"{bd.qed_contribution:.4f}" if bd.qed_contribution is not None else "")
                else:
                    # No breakdown (failed molecule)
                    for _ in target_names:
                        row.extend(["", "", "", ""])
                    row.extend(["", ""])

                writer.writerow(row)

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def __enter__(self) -> "RewardCalculator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
