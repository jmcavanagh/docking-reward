"""Vina docking wrapper."""

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from meeko import PDBQTMolecule
from rdkit import Chem
from vina import Vina

from .config import TargetConfig, VinaConfig

logger = logging.getLogger(__name__)


@dataclass
class DockingResult:
    """Result of a single docking run."""

    target_name: str
    scores: list[float] = field(default_factory=list)  # Vina scores (kcal/mol, negative)
    poses: list[Chem.Mol] = field(default_factory=list)  # RDKit Mol objects
    best_score: float = 0.0  # Best (most negative) Vina score
    success: bool = False
    error: Optional[str] = None


def pdbqt_to_mol(pdbqt_string: str) -> Optional[Chem.Mol]:
    """
    Convert PDBQT string to RDKit Mol.

    Args:
        pdbqt_string: PDBQT format string

    Returns:
        RDKit Mol object or None if conversion fails
    """
    try:
        # Use Meeko to parse PDBQT
        pdbqt_mol = PDBQTMolecule(pdbqt_string, is_dlg=False, skip_typing=True)

        # Get RDKit molecule from Meeko
        # PDBQTMolecule can export to RDKit mol
        mol = pdbqt_mol.export_rdkit_mol()

        if mol is not None:
            return mol

    except Exception as e:
        logger.debug(f"Meeko conversion failed: {e}")

    # Fallback: try parsing as PDB (strip PDBQT-specific columns)
    try:
        pdb_lines = []
        for line in pdbqt_string.split("\n"):
            if line.startswith(("ATOM", "HETATM")):
                # Keep only first 66 characters (standard PDB format)
                pdb_lines.append(line[:66])
            elif line.startswith(("TER", "END", "MODEL", "ENDMDL")):
                pdb_lines.append(line)

        pdb_string = "\n".join(pdb_lines)
        mol = Chem.MolFromPDBBlock(pdb_string, removeHs=False, sanitize=False)

        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass  # Keep unsanitized mol
            return mol

    except Exception as e:
        logger.debug(f"PDB fallback conversion failed: {e}")

    return None


def dock_ligand(
    ligand_pdbqt: Path,
    protein_pdbqt: Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    exhaustiveness: int = 8,
    n_poses: int = 9,
    energy_range: float = 3.0,
    seed: Optional[int] = None,
    target_name: str = "unknown",
) -> DockingResult:
    """
    Dock a ligand to a protein using Vina.

    Args:
        ligand_pdbqt: Path to prepared ligand PDBQT
        protein_pdbqt: Path to prepared protein PDBQT
        center: Docking box center (x, y, z) in Angstroms
        size: Docking box size (x, y, z) in Angstroms
        exhaustiveness: Vina exhaustiveness parameter
        n_poses: Number of poses to return
        energy_range: Max energy difference from best pose
        seed: Random seed for reproducibility
        target_name: Name of target for result tracking

    Returns:
        DockingResult with scores and poses
    """
    result = DockingResult(target_name=target_name)

    try:
        # Initialize Vina
        v = Vina(sf_name="vina", verbosity=0)

        # Set receptor
        v.set_receptor(str(protein_pdbqt))

        # Set ligand
        v.set_ligand_from_file(str(ligand_pdbqt))

        # Set search box
        v.compute_vina_maps(
            center=list(center),
            box_size=list(size),
        )

        # Run docking
        v.dock(
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            min_rmsd=1.0,
            max_evals=0,
        )

        # Get energies
        energies = v.energies()
        if energies is None or len(energies) == 0:
            result.error = "No poses generated"
            return result

        # Extract scores (first column is total score)
        result.scores = [float(e[0]) for e in energies]
        result.best_score = min(result.scores)

        # Get poses as PDBQT strings and convert to RDKit Mol
        with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False, mode="w") as f:
            temp_path = Path(f.name)

        try:
            v.write_poses(str(temp_path), n_poses=n_poses, overwrite=True)

            with open(temp_path) as f:
                pdbqt_content = f.read()

            # Split into individual poses (separated by MODEL/ENDMDL)
            pose_blocks = []
            current_block = []

            for line in pdbqt_content.split("\n"):
                if line.startswith("MODEL"):
                    current_block = []
                elif line.startswith("ENDMDL"):
                    if current_block:
                        pose_blocks.append("\n".join(current_block))
                else:
                    current_block.append(line)

            # If no MODEL/ENDMDL markers, treat whole file as one pose
            if not pose_blocks and current_block:
                pose_blocks = ["\n".join(current_block)]
            elif not pose_blocks:
                pose_blocks = [pdbqt_content]

            # Convert each pose to RDKit Mol
            for pose_block in pose_blocks:
                mol = pdbqt_to_mol(pose_block)
                if mol is not None:
                    result.poses.append(mol)

        finally:
            temp_path.unlink(missing_ok=True)

        result.success = True

    except Exception as e:
        result.error = str(e)
        logger.warning(f"Docking failed for target '{target_name}': {e}")

    return result


def dock_to_targets(
    ligand_pdbqt: Path,
    targets: list[TargetConfig],
    protein_pdbqts: dict[str, Path],
    global_vina_config: VinaConfig,
) -> dict[str, DockingResult]:
    """
    Dock a ligand to all configured targets.

    Args:
        ligand_pdbqt: Path to prepared ligand
        targets: List of target configurations
        protein_pdbqts: Dict mapping target name to PDBQT path
        global_vina_config: Global Vina settings

    Returns:
        Dict mapping target name to DockingResult
    """
    results = {}

    for target in targets:
        protein_pdbqt = protein_pdbqts.get(target.name)
        if protein_pdbqt is None:
            results[target.name] = DockingResult(
                target_name=target.name,
                success=False,
                error=f"No prepared PDBQT for target '{target.name}'",
            )
            continue

        # Use per-target settings if specified, otherwise use global
        exhaustiveness = (
            target.exhaustiveness
            if target.exhaustiveness is not None
            else global_vina_config.exhaustiveness
        )
        n_poses = (
            target.n_poses
            if target.n_poses is not None
            else global_vina_config.n_poses
        )
        energy_range = (
            target.energy_range
            if target.energy_range is not None
            else global_vina_config.energy_range
        )

        results[target.name] = dock_ligand(
            ligand_pdbqt=ligand_pdbqt,
            protein_pdbqt=protein_pdbqt,
            center=target.center,
            size=target.size,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            energy_range=energy_range,
            seed=global_vina_config.seed,
            target_name=target.name,
        )

    return results
