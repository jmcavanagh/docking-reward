"""Uni-Dock GPU docking wrapper for batched ligand docking."""

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import TargetConfig, VinaConfig

logger = logging.getLogger(__name__)

# Scoring functions supported by Uni-Dock
UNIDOCK_SCORING_FUNCTIONS = frozenset({
    "vina",        # Standard Vina scoring
    "vinardo",     # Vinardo scoring (faster, similar accuracy)
    "ad4",         # AutoDock4 scoring function
})


@dataclass
class UnidockResult:
    """Result of a batched Uni-Dock run for one target."""

    target_name: str
    # Maps ligand index to list of scores
    scores_by_ligand: dict[int, list[float]] = field(default_factory=dict)
    # Maps ligand index to list of pose PDBQT strings
    poses_by_ligand: dict[int, list[str]] = field(default_factory=dict)
    # Maps ligand index to best score
    best_scores: dict[int, float] = field(default_factory=dict)
    # Maps ligand index to error message (if failed)
    errors: dict[int, str] = field(default_factory=dict)
    success: bool = False
    error: Optional[str] = None


def check_unidock_available() -> bool:
    """Check if Uni-Dock is installed and available."""
    return shutil.which("unidock") is not None


def run_unidock_batch(
    ligand_pdbqts: list[Path],
    protein_pdbqt: Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    output_dir: Path,
    exhaustiveness: int = 8,
    n_poses: int = 9,
    energy_range: float = 3.0,
    seed: Optional[int] = None,
    scoring_function: str = "vina",
    target_name: str = "unknown",
) -> UnidockResult:
    """
    Run Uni-Dock on a batch of ligands against a single target.

    This is much faster than running Vina sequentially because Uni-Dock
    processes all ligands in parallel on the GPU.

    Args:
        ligand_pdbqts: List of paths to prepared ligand PDBQT files
        protein_pdbqt: Path to prepared protein PDBQT
        center: Docking box center (x, y, z) in Angstroms
        size: Docking box size (x, y, z) in Angstroms
        output_dir: Directory for output pose files
        exhaustiveness: Search exhaustiveness
        n_poses: Number of poses per ligand
        energy_range: Max energy difference from best pose
        seed: Random seed for reproducibility
        scoring_function: Scoring function ("vina", "vinardo", "ad4")
        target_name: Name of target for result tracking

    Returns:
        UnidockResult with scores and poses for all ligands
    """
    result = UnidockResult(target_name=target_name)

    if not ligand_pdbqts:
        result.error = "No ligands provided"
        return result

    if scoring_function not in UNIDOCK_SCORING_FUNCTIONS:
        result.error = f"Invalid scoring function '{scoring_function}'. Must be one of: {', '.join(UNIDOCK_SCORING_FUNCTIONS)}"
        return result

    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create a ligand index file for Uni-Dock batch mode
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as ligand_list_file:
            ligand_list_path = Path(ligand_list_file.name)
            for pdbqt_path in ligand_pdbqts:
                ligand_list_file.write(f"{pdbqt_path}\n")

        # Build Uni-Dock command
        cmd = [
            "unidock",
            "--receptor", str(protein_pdbqt),
            "--ligand_index", str(ligand_list_path),
            "--center_x", str(center[0]),
            "--center_y", str(center[1]),
            "--center_z", str(center[2]),
            "--size_x", str(size[0]),
            "--size_y", str(size[1]),
            "--size_z", str(size[2]),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", str(n_poses),
            "--energy_range", str(energy_range),
            "--scoring", scoring_function,
            "--dir", str(output_dir),
        ]

        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        logger.info(f"Running Uni-Dock on {len(ligand_pdbqts)} ligands for target '{target_name}'")
        logger.debug(f"Command: {' '.join(cmd)}")

        # Run Uni-Dock
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout for large batches
        )

        if proc.returncode != 0:
            result.error = f"Uni-Dock failed: {proc.stderr}"
            logger.error(f"Uni-Dock stderr: {proc.stderr}")
            return result

        # Parse results from output directory
        # Uni-Dock writes output files as <ligand_name>_out.pdbqt
        for idx, ligand_path in enumerate(ligand_pdbqts):
            ligand_stem = ligand_path.stem
            output_path = output_dir / f"{ligand_stem}_out.pdbqt"

            if not output_path.exists():
                result.errors[idx] = f"No output file for ligand {ligand_stem}"
                continue

            try:
                scores, poses = parse_unidock_output(output_path)
                if scores:
                    result.scores_by_ligand[idx] = scores
                    result.poses_by_ligand[idx] = poses
                    result.best_scores[idx] = min(scores)
                else:
                    result.errors[idx] = "No poses in output"
            except Exception as e:
                result.errors[idx] = f"Error parsing output: {e}"

        result.success = True

    except subprocess.TimeoutExpired:
        result.error = "Uni-Dock timed out"
    except FileNotFoundError:
        result.error = "Uni-Dock not found. Install with: conda install -c conda-forge unidock"
    except Exception as e:
        result.error = f"Uni-Dock error: {e}"
        logger.exception(f"Uni-Dock batch docking failed: {e}")
    finally:
        # Clean up ligand list file
        if "ligand_list_path" in locals():
            ligand_list_path.unlink(missing_ok=True)

    return result


def parse_unidock_output(output_path: Path) -> tuple[list[float], list[str]]:
    """
    Parse Uni-Dock output PDBQT file to extract scores and poses.

    Args:
        output_path: Path to output PDBQT file from Uni-Dock

    Returns:
        Tuple of (scores list, pose PDBQT strings list)
    """
    scores = []
    poses = []
    current_pose = []
    current_score = None

    with open(output_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                current_pose = []
                current_score = None
            elif line.startswith("ENDMDL"):
                if current_pose and current_score is not None:
                    poses.append("\n".join(current_pose))
                    scores.append(current_score)
            elif line.startswith("REMARK VINA RESULT:") or line.startswith("REMARK RESULT:"):
                # Parse score from REMARK line
                # Format: REMARK VINA RESULT:    -7.5      0.000      0.000
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        current_score = float(parts[3])
                    except (ValueError, IndexError):
                        pass
            else:
                current_pose.append(line.rstrip())

    # Handle case where file has no MODEL/ENDMDL markers
    if not poses and current_pose and current_score is not None:
        poses.append("\n".join(current_pose))
        scores.append(current_score)

    return scores, poses


def dock_batch_to_target(
    ligand_pdbqts: list[Path],
    target: TargetConfig,
    protein_pdbqt: Path,
    output_dir: Path,
    global_config: VinaConfig,
    scoring_function: str = "vina",
) -> UnidockResult:
    """
    Dock a batch of ligands to a single target using Uni-Dock.

    Args:
        ligand_pdbqts: List of ligand PDBQT paths
        target: Target configuration
        protein_pdbqt: Path to prepared protein PDBQT
        output_dir: Directory for output files
        global_config: Global Vina/docking config
        scoring_function: Scoring function to use

    Returns:
        UnidockResult with all docking results
    """
    # Use per-target settings if specified, otherwise use global
    exhaustiveness = (
        target.exhaustiveness
        if target.exhaustiveness is not None
        else global_config.exhaustiveness
    )
    n_poses = (
        target.n_poses
        if target.n_poses is not None
        else global_config.n_poses
    )
    energy_range = (
        target.energy_range
        if target.energy_range is not None
        else global_config.energy_range
    )

    return run_unidock_batch(
        ligand_pdbqts=ligand_pdbqts,
        protein_pdbqt=protein_pdbqt,
        center=target.center,
        size=target.size,
        output_dir=output_dir / target.name,
        exhaustiveness=exhaustiveness,
        n_poses=n_poses,
        energy_range=energy_range,
        seed=global_config.seed,
        scoring_function=scoring_function,
        target_name=target.name,
    )
