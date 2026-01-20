"""Uni-Dock GPU docking wrapper for batched ligand docking."""

import logging
import os
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import TargetConfig, VinaConfig
from .ligand_prep import validate_pdbqt_for_unidock

logger = logging.getLogger(__name__)

# Maximum retry attempts for batch docking
MAX_BATCH_RETRIES = 3

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


def filter_ligands_for_unidock(
    ligand_pdbqts: list[Path],
) -> tuple[list[Path], dict[int, str]]:
    """
    Pre-filter ligands for Uni-Dock compatibility.

    Args:
        ligand_pdbqts: List of paths to prepared ligand PDBQT files

    Returns:
        Tuple of (valid_ligands, errors_by_original_index)
    """
    valid_ligands = []
    errors = {}

    for idx, pdbqt_path in enumerate(ligand_pdbqts):
        is_valid, error_msg = validate_pdbqt_for_unidock(pdbqt_path)
        if is_valid:
            valid_ligands.append(pdbqt_path)
        else:
            errors[idx] = f"Pre-filter rejected: {error_msg}"
            logger.debug(f"Ligand {pdbqt_path.stem} rejected: {error_msg}")

    if errors:
        logger.info(
            f"Pre-filtered {len(errors)} incompatible ligands "
            f"({len(valid_ligands)} remaining)"
        )

    return valid_ligands, errors


def parse_unidock_error(stderr: str) -> list[str]:
    """
    Parse Uni-Dock stderr to identify problematic ligand files.

    Args:
        stderr: Uni-Dock stderr output

    Returns:
        List of ligand filenames that caused errors
    """
    bad_ligands = []

    # Common error patterns in Uni-Dock output
    # Pattern 1: "ligand mol_123.pdbqt has atom type X"
    atom_type_pattern = re.compile(r"ligand\s+(\S+\.pdbqt)\s+has\s+atom\s+type", re.IGNORECASE)

    # Pattern 2: "mol_123.pdbqt: torsion"
    torsion_pattern = re.compile(r"(\S+\.pdbqt):\s*torsion", re.IGNORECASE)

    # Pattern 3: General file reference in error context
    file_error_pattern = re.compile(r"error.*?(\bmol_\d+\.pdbqt\b)", re.IGNORECASE)

    # Pattern 4: "Too many torsion" with file reference
    too_many_torsion_pattern = re.compile(r"(\S+\.pdbqt).*?too\s+many\s+torsion", re.IGNORECASE)

    for pattern in [atom_type_pattern, torsion_pattern, file_error_pattern, too_many_torsion_pattern]:
        matches = pattern.findall(stderr)
        for match in matches:
            filename = match if match.endswith(".pdbqt") else f"{match}.pdbqt"
            if filename not in bad_ligands:
                bad_ligands.append(filename)

    return bad_ligands


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
    prefilter: bool = True,
    retry_on_failure: bool = True,
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
        prefilter: Pre-filter ligands for compatibility (default True)
        retry_on_failure: Retry with bad ligands removed on failure (default True)

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

    # Build mapping from original indices to paths
    original_indices = {path: idx for idx, path in enumerate(ligand_pdbqts)}

    # Pre-filter ligands if enabled
    working_ligands = list(ligand_pdbqts)
    if prefilter:
        working_ligands, prefilter_errors = filter_ligands_for_unidock(ligand_pdbqts)
        result.errors.update(prefilter_errors)

        if not working_ligands:
            result.error = "All ligands rejected during pre-filtering"
            return result

    # Track ligands excluded during retries
    excluded_ligands: set[Path] = set()
    attempt = 0

    while attempt < MAX_BATCH_RETRIES:
        attempt += 1

        # Remove excluded ligands from working set
        current_ligands = [l for l in working_ligands if l not in excluded_ligands]

        if not current_ligands:
            result.error = "No valid ligands remaining after filtering"
            break

        logger.info(
            f"Uni-Dock attempt {attempt}/{MAX_BATCH_RETRIES} "
            f"with {len(current_ligands)} ligands for target '{target_name}'"
        )

        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Build Uni-Dock command
            # Use --gpu_batch with ligand paths directly for true GPU batch processing
            cmd = [
                "unidock",
                "--receptor", str(protein_pdbqt),
                "--gpu_batch",
            ]
            # Add all ligand paths as separate arguments
            cmd.extend([str(p) for p in current_ligands])

            cmd.extend([
                "--center_x", str(center[0]),
                "--center_y", str(center[1]),
                "--center_z", str(center[2]),
                "--size_x", str(size[0]),
                "--size_y", str(size[1]),
                "--size_z", str(size[2]),
                "--exhaustiveness", str(exhaustiveness),
                "--max_step", str(n_poses),
                "--energy_range", str(energy_range),
                "--scoring", scoring_function,
                "--dir", str(output_dir),
            ])

            if seed is not None:
                cmd.extend(["--seed", str(seed)])

            logger.debug(f"Command: {' '.join(cmd)}")

            # Run Uni-Dock
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout for large batches
            )

            if proc.returncode != 0:
                stderr = proc.stderr or ""
                logger.warning(f"Uni-Dock failed (attempt {attempt}): {stderr[:500]}")

                if retry_on_failure and attempt < MAX_BATCH_RETRIES:
                    # Try to identify and exclude bad ligands
                    bad_filenames = parse_unidock_error(stderr)

                    if bad_filenames:
                        for ligand_path in current_ligands:
                            if ligand_path.name in bad_filenames:
                                excluded_ligands.add(ligand_path)
                                orig_idx = original_indices.get(ligand_path)
                                if orig_idx is not None:
                                    result.errors[orig_idx] = f"Excluded after batch failure: {stderr[:200]}"
                                logger.info(f"Excluding problematic ligand: {ligand_path.name}")

                        if excluded_ligands:
                            logger.info(
                                f"Retrying without {len(excluded_ligands)} problematic ligands"
                            )
                            continue
                    else:
                        # Couldn't identify specific bad ligands
                        logger.warning("Could not identify problematic ligands from error output")

                # No retry or couldn't identify bad ligands
                result.error = f"Uni-Dock failed: {stderr}"
                logger.error(f"Uni-Dock stderr: {stderr}")
                break

            # Success! Parse results from output directory
            for ligand_path in current_ligands:
                ligand_stem = ligand_path.stem
                output_path = output_dir / f"{ligand_stem}_out.pdbqt"
                orig_idx = original_indices.get(ligand_path)

                if orig_idx is None:
                    continue

                if not output_path.exists():
                    result.errors[orig_idx] = f"No output file for ligand {ligand_stem}"
                    continue

                try:
                    scores, poses = parse_unidock_output(output_path)
                    if scores:
                        result.scores_by_ligand[orig_idx] = scores
                        result.poses_by_ligand[orig_idx] = poses
                        result.best_scores[orig_idx] = min(scores)
                    else:
                        result.errors[orig_idx] = "No poses in output"
                except Exception as e:
                    result.errors[orig_idx] = f"Error parsing output: {e}"

            result.success = True
            break  # Success, exit retry loop

        except subprocess.TimeoutExpired:
            result.error = "Uni-Dock timed out"
            break
        except FileNotFoundError:
            result.error = "Uni-Dock not found. Install with: conda install -c conda-forge unidock"
            break
        except Exception as e:
            result.error = f"Uni-Dock error: {e}"
            logger.exception(f"Uni-Dock batch docking failed: {e}")
            break

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


def _run_gpu_batch(
    gpu_id: int,
    ligand_pdbqts: list[Path],
    protein_pdbqt: Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    output_dir: Path,
    exhaustiveness: int,
    n_poses: int,
    energy_range: float,
    seed: Optional[int],
    scoring_function: str,
    target_name: str,
) -> UnidockResult:
    """
    Run Uni-Dock on a specific GPU.

    This is a wrapper for run_unidock_batch that sets CUDA_VISIBLE_DEVICES.

    Args:
        gpu_id: GPU index to use (0, 1, 2, ...)
        Other args: Same as run_unidock_batch

    Returns:
        UnidockResult from this GPU batch
    """
    # Set environment to use only the specified GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logger.info(f"Starting Uni-Dock on GPU {gpu_id} with {len(ligand_pdbqts)} ligands")

    # Create GPU-specific output directory
    gpu_output_dir = output_dir / f"gpu_{gpu_id}"

    result = UnidockResult(target_name=target_name)

    if not ligand_pdbqts:
        result.error = "No ligands provided"
        return result

    # Build mapping from original indices to paths
    original_indices = {path: idx for idx, path in enumerate(ligand_pdbqts)}

    # Pre-filter ligands
    working_ligands, prefilter_errors = filter_ligands_for_unidock(ligand_pdbqts)
    result.errors.update(prefilter_errors)

    if not working_ligands:
        result.error = "All ligands rejected during pre-filtering"
        return result

    try:
        gpu_output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "unidock",
            "--receptor", str(protein_pdbqt),
            "--gpu_batch",
        ]
        cmd.extend([str(p) for p in working_ligands])
        cmd.extend([
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
            "--dir", str(gpu_output_dir),
        ])

        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        logger.debug(f"GPU {gpu_id} command: {' '.join(cmd[:10])}...")

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            env=env,  # Use modified environment with CUDA_VISIBLE_DEVICES
        )

        if proc.returncode != 0:
            result.error = f"Uni-Dock failed on GPU {gpu_id}: {proc.stderr[:500]}"
            logger.error(f"GPU {gpu_id} stderr: {proc.stderr}")
            return result

        # Parse results
        for ligand_path in working_ligands:
            ligand_stem = ligand_path.stem
            output_path = gpu_output_dir / f"{ligand_stem}_out.pdbqt"
            orig_idx = original_indices.get(ligand_path)

            if orig_idx is None:
                continue

            if not output_path.exists():
                result.errors[orig_idx] = f"No output file for ligand {ligand_stem}"
                continue

            try:
                scores, poses = parse_unidock_output(output_path)
                if scores:
                    result.scores_by_ligand[orig_idx] = scores
                    result.poses_by_ligand[orig_idx] = poses
                    result.best_scores[orig_idx] = min(scores)
                else:
                    result.errors[orig_idx] = "No poses in output"
            except Exception as e:
                result.errors[orig_idx] = f"Error parsing output: {e}"

        result.success = True
        logger.info(f"GPU {gpu_id} completed: {len(result.best_scores)} ligands docked")

    except subprocess.TimeoutExpired:
        result.error = f"Uni-Dock timed out on GPU {gpu_id}"
    except FileNotFoundError:
        result.error = "Uni-Dock not found"
    except Exception as e:
        result.error = f"Uni-Dock error on GPU {gpu_id}: {e}"
        logger.exception(f"GPU {gpu_id} docking failed: {e}")

    return result


def run_unidock_multi_gpu(
    ligand_pdbqts: list[Path],
    protein_pdbqt: Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    output_dir: Path,
    n_gpus: int,
    exhaustiveness: int = 8,
    n_poses: int = 9,
    energy_range: float = 3.0,
    seed: Optional[int] = None,
    scoring_function: str = "vina",
    target_name: str = "unknown",
) -> UnidockResult:
    """
    Run Uni-Dock across multiple GPUs in parallel.

    Splits the ligand batch evenly across GPUs and runs them concurrently.

    Args:
        ligand_pdbqts: List of paths to prepared ligand PDBQT files
        protein_pdbqt: Path to prepared protein PDBQT
        center: Docking box center (x, y, z) in Angstroms
        size: Docking box size (x, y, z) in Angstroms
        output_dir: Directory for output pose files
        n_gpus: Number of GPUs to use
        exhaustiveness: Search exhaustiveness
        n_poses: Number of poses per ligand
        energy_range: Max energy difference from best pose
        seed: Random seed for reproducibility
        scoring_function: Scoring function ("vina", "vinardo", "ad4")
        target_name: Name of target for result tracking

    Returns:
        Merged UnidockResult with scores and poses from all GPUs
    """
    if n_gpus <= 1:
        # Fall back to single GPU
        return run_unidock_batch(
            ligand_pdbqts=ligand_pdbqts,
            protein_pdbqt=protein_pdbqt,
            center=center,
            size=size,
            output_dir=output_dir,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            energy_range=energy_range,
            seed=seed,
            scoring_function=scoring_function,
            target_name=target_name,
        )

    # Build mapping from path to original index
    path_to_orig_idx = {path: idx for idx, path in enumerate(ligand_pdbqts)}

    # Split ligands across GPUs
    n_ligands = len(ligand_pdbqts)
    batch_size = (n_ligands + n_gpus - 1) // n_gpus  # Ceiling division

    gpu_batches = []
    for gpu_id in range(n_gpus):
        start_idx = gpu_id * batch_size
        end_idx = min(start_idx + batch_size, n_ligands)
        if start_idx < n_ligands:
            gpu_batches.append((gpu_id, ligand_pdbqts[start_idx:end_idx]))

    logger.info(
        f"Splitting {n_ligands} ligands across {len(gpu_batches)} GPUs "
        f"(~{batch_size} per GPU)"
    )

    # Run GPU batches in parallel using ProcessPoolExecutor
    merged_result = UnidockResult(target_name=target_name)
    gpu_results = []

    with ProcessPoolExecutor(max_workers=len(gpu_batches)) as executor:
        futures = {}
        for gpu_id, batch_ligands in gpu_batches:
            future = executor.submit(
                _run_gpu_batch,
                gpu_id=gpu_id,
                ligand_pdbqts=batch_ligands,
                protein_pdbqt=protein_pdbqt,
                center=center,
                size=size,
                output_dir=output_dir,
                exhaustiveness=exhaustiveness,
                n_poses=n_poses,
                energy_range=energy_range,
                seed=seed,
                scoring_function=scoring_function,
                target_name=target_name,
            )
            futures[future] = (gpu_id, batch_ligands)

        for future in as_completed(futures):
            gpu_id, batch_ligands = futures[future]
            try:
                gpu_result = future.result()
                gpu_results.append((gpu_id, batch_ligands, gpu_result))
            except Exception as e:
                logger.error(f"GPU {gpu_id} batch failed: {e}")
                # Mark all ligands in this batch as failed
                for ligand_path in batch_ligands:
                    orig_idx = path_to_orig_idx.get(ligand_path)
                    if orig_idx is not None:
                        merged_result.errors[orig_idx] = f"GPU {gpu_id} batch error: {e}"

    # Merge results from all GPUs
    any_success = False
    for gpu_id, batch_ligands, gpu_result in gpu_results:
        if gpu_result.success:
            any_success = True

        # Map batch-local indices back to original indices
        for batch_idx, ligand_path in enumerate(batch_ligands):
            orig_idx = path_to_orig_idx.get(ligand_path)
            if orig_idx is None:
                continue

            if batch_idx in gpu_result.scores_by_ligand:
                merged_result.scores_by_ligand[orig_idx] = gpu_result.scores_by_ligand[batch_idx]
                merged_result.poses_by_ligand[orig_idx] = gpu_result.poses_by_ligand[batch_idx]
                merged_result.best_scores[orig_idx] = gpu_result.best_scores[batch_idx]
            elif batch_idx in gpu_result.errors:
                merged_result.errors[orig_idx] = gpu_result.errors[batch_idx]

        if gpu_result.error and not gpu_result.success:
            # Record GPU-level errors
            for ligand_path in batch_ligands:
                orig_idx = path_to_orig_idx.get(ligand_path)
                if orig_idx is not None and orig_idx not in merged_result.errors:
                    merged_result.errors[orig_idx] = gpu_result.error

    merged_result.success = any_success
    if not any_success:
        merged_result.error = "All GPU batches failed"

    logger.info(
        f"Multi-GPU docking complete: {len(merged_result.best_scores)} succeeded, "
        f"{len(merged_result.errors)} failed"
    )

    return merged_result


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

    Uses multi-GPU parallelism if global_config.n_gpus > 1.

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

    # Use multi-GPU if configured
    n_gpus = getattr(global_config, "n_gpus", 1)
    if n_gpus > 1:
        return run_unidock_multi_gpu(
            ligand_pdbqts=ligand_pdbqts,
            protein_pdbqt=protein_pdbqt,
            center=target.center,
            size=target.size,
            output_dir=output_dir / target.name,
            n_gpus=n_gpus,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            energy_range=energy_range,
            seed=global_config.seed,
            scoring_function=scoring_function,
            target_name=target.name,
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
