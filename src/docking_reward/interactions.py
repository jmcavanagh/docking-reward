"""Protein-ligand interaction analysis using PLIP."""

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rdkit import Chem

from .config import InteractionConfig

logger = logging.getLogger(__name__)


@dataclass
class InteractionCount:
    """Counts of different interaction types with a residue."""

    residue: str  # e.g., "ASP123" or "A:ASP:123"
    hydrogen_bonds: int = 0
    hydrophobic: int = 0
    salt_bridges: int = 0
    pi_stacking: int = 0

    @property
    def any(self) -> int:
        """Total count of all interactions."""
        return self.hydrogen_bonds + self.hydrophobic + self.salt_bridges + self.pi_stacking

    def get_count(self, interaction_type: str) -> int:
        """Get count for a specific interaction type."""
        if interaction_type == "hydrogen_bond":
            return self.hydrogen_bonds
        elif interaction_type == "hydrophobic":
            return self.hydrophobic
        elif interaction_type == "salt_bridge":
            return self.salt_bridges
        elif interaction_type == "pi_stacking":
            return self.pi_stacking
        elif interaction_type == "any":
            return self.any
        else:
            logger.warning(f"Unknown interaction type: {interaction_type}")
            return 0


def _normalize_residue_id(residue_str: str) -> str:
    """
    Normalize residue identifier for matching.

    Handles formats like:
    - "ASP123" -> "ASP123"
    - "A:ASP:123" -> "ASP123"
    - "ASP:123:A" -> "ASP123"

    Args:
        residue_str: Residue identifier string

    Returns:
        Normalized residue ID (e.g., "ASP123")
    """
    # Remove chain identifiers and colons
    parts = residue_str.replace(":", " ").split()

    res_name = None
    res_num = None

    for part in parts:
        # Check if it's a residue name (3 letters)
        if len(part) == 3 and part.isalpha():
            res_name = part.upper()
        # Check if it's a number (possibly with insertion code)
        elif part[0].isdigit() or (len(part) > 1 and part[:-1].isdigit()):
            res_num = part
        # Could be combined like "ASP123"
        elif len(part) > 3:
            # Try to split into name + number
            for i, char in enumerate(part):
                if char.isdigit():
                    res_name = part[:i].upper()
                    res_num = part[i:]
                    break

    if res_name and res_num:
        return f"{res_name}{res_num}"
    return residue_str.upper()


def _create_complex_pdb(
    ligand_mol: Chem.Mol, protein_pdb: Path, output_path: Path
) -> bool:
    """
    Create a complex PDB file with protein and ligand for PLIP analysis.

    Args:
        ligand_mol: RDKit Mol of docked ligand
        protein_pdb: Path to protein PDB file
        output_path: Where to write the complex PDB

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get ligand PDB block
        ligand_pdb = Chem.MolToPDBBlock(ligand_mol)
        if not ligand_pdb:
            return False

        # Read protein PDB
        with open(protein_pdb) as f:
            protein_lines = f.readlines()

        # Write combined file
        with open(output_path, "w") as f:
            # Write protein atoms (skip END)
            for line in protein_lines:
                if not line.startswith("END"):
                    f.write(line)

            # Write TER between protein and ligand
            f.write("TER\n")

            # Write ligand as HETATM records with residue name "LIG"
            for line in ligand_pdb.split("\n"):
                if line.startswith(("ATOM", "HETATM")):
                    # Convert to HETATM and set residue name to LIG
                    new_line = "HETATM" + line[6:17] + "LIG" + line[20:]
                    f.write(new_line + "\n")
                elif line.startswith("CONECT"):
                    f.write(line + "\n")

            f.write("END\n")

        return True

    except Exception as e:
        logger.warning(f"Error creating complex PDB: {e}")
        return False


def analyze_interactions(
    ligand_mol: Chem.Mol, protein_pdb: Path
) -> list[InteractionCount]:
    """
    Analyze protein-ligand interactions using PLIP.

    Args:
        ligand_mol: RDKit Mol of docked pose
        protein_pdb: Path to protein PDB file

    Returns:
        List of InteractionCount objects, one per interacting residue
    """
    try:
        from plip.structure.preparation import PDBComplex
    except ImportError as e:
        logger.error(f"PLIP not available: {e}")
        return []

    # Create temporary complex PDB
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
        temp_complex = Path(f.name)

    try:
        if not _create_complex_pdb(ligand_mol, protein_pdb, temp_complex):
            return []

        # Run PLIP analysis
        complex_obj = PDBComplex()
        complex_obj.load_pdb(str(temp_complex))

        # Analyze interactions with ligand
        complex_obj.analyze()

        # Collect interaction counts by residue
        residue_interactions: dict[str, InteractionCount] = {}

        for ligand_id, ligand_data in complex_obj.interaction_sets.items():
            # Process hydrogen bonds
            for hbond in ligand_data.hbonds_ldon + ligand_data.hbonds_pdon:
                res_id = _normalize_residue_id(f"{hbond.restype}{hbond.resnr}")
                if res_id not in residue_interactions:
                    residue_interactions[res_id] = InteractionCount(residue=res_id)
                residue_interactions[res_id].hydrogen_bonds += 1

            # Process hydrophobic contacts
            for hydrophobic in ligand_data.hydrophobic_contacts:
                res_id = _normalize_residue_id(f"{hydrophobic.restype}{hydrophobic.resnr}")
                if res_id not in residue_interactions:
                    residue_interactions[res_id] = InteractionCount(residue=res_id)
                residue_interactions[res_id].hydrophobic += 1

            # Process salt bridges
            for saltbridge in ligand_data.saltbridge_lneg + ligand_data.saltbridge_pneg:
                res_id = _normalize_residue_id(f"{saltbridge.restype}{saltbridge.resnr}")
                if res_id not in residue_interactions:
                    residue_interactions[res_id] = InteractionCount(residue=res_id)
                residue_interactions[res_id].salt_bridges += 1

            # Process pi-stacking
            for pistack in ligand_data.pistacking:
                res_id = _normalize_residue_id(f"{pistack.restype}{pistack.resnr}")
                if res_id not in residue_interactions:
                    residue_interactions[res_id] = InteractionCount(residue=res_id)
                residue_interactions[res_id].pi_stacking += 1

        return list(residue_interactions.values())

    except Exception as e:
        logger.warning(f"PLIP analysis failed: {e}")
        return []

    finally:
        temp_complex.unlink(missing_ok=True)


def score_interactions(
    interaction_counts: list[InteractionCount],
    interaction_configs: list[InteractionConfig],
) -> float:
    """
    Calculate interaction-based score contribution.

    Args:
        interaction_counts: Detected interactions from PLIP
        interaction_configs: Configured interaction weights

    Returns:
        Total interaction score (sum of weight * count for each configured interaction)
    """
    total_score = 0.0

    # Build lookup by normalized residue ID
    counts_by_residue = {
        _normalize_residue_id(ic.residue): ic for ic in interaction_counts
    }

    for config in interaction_configs:
        normalized_residue = _normalize_residue_id(config.residue)

        if normalized_residue in counts_by_residue:
            count = counts_by_residue[normalized_residue].get_count(config.type)
            contribution = config.weight * count
            total_score += contribution

            if count > 0:
                logger.debug(
                    f"Interaction score: {config.residue} {config.type} "
                    f"count={count}, weight={config.weight}, contribution={contribution}"
                )

    return total_score
