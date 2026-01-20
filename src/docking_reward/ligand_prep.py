"""Ligand preparation: SMILES to 3D conformer to PDBQT."""

import logging
from pathlib import Path
from typing import Optional

from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

# Default pH for protonation (physiological pH)
DEFAULT_PH = 7.4
# pH range for Dimorphite-DL (centered on target pH)
PH_RANGE = 0.5


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    Parse SMILES string to RDKit Mol object.

    Args:
        smiles: SMILES string

    Returns:
        RDKit Mol object, or None if parsing fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Failed to parse SMILES: {smiles}")
            return None
        return mol
    except Exception as e:
        logger.warning(f"Error parsing SMILES '{smiles}': {e}")
        return None


def protonate_smiles(smiles: str, ph: float = DEFAULT_PH) -> str:
    """
    Protonate a SMILES string for a target pH using Dimorphite-DL.

    This ensures correct protonation states for docking:
    - Carboxylic acids (-COOH) become carboxylates (-COO⁻) at pH 7.4
    - Primary amines become protonated (-NH3⁺) at pH 7.4
    - etc.

    Args:
        smiles: Input SMILES string
        ph: Target pH (default 7.4)

    Returns:
        Protonated SMILES string (returns original if protonation fails)
    """
    try:
        from dimorphite_dl import protonate_smiles as dimorphite_protonate

        # Dimorphite-DL returns a list of possible protonation states
        # We use a narrow pH range to get the dominant form
        protonated = dimorphite_protonate(
            smiles,
            ph_min=ph - PH_RANGE,
            ph_max=ph + PH_RANGE,
        )

        if protonated:
            # Return the first (most likely) protonation state
            result = protonated[0]
            logger.debug(f"Protonated {smiles} -> {result} at pH {ph}")
            return result
        else:
            logger.debug(f"Dimorphite-DL returned no results for {smiles}, using original")
            return smiles

    except ImportError:
        logger.warning(
            "dimorphite-dl not installed. Install with: pip install dimorphite-dl. "
            "Using original SMILES without pH-dependent protonation."
        )
        return smiles
    except Exception as e:
        logger.debug(f"Protonation failed for {smiles}: {e}, using original")
        return smiles


def embed_molecule(mol: Chem.Mol, n_conformers: int = 1, random_seed: int = 42) -> Optional[Chem.Mol]:
    """
    Generate 3D conformer(s) for a molecule using ETKDG.

    Assumes biological pH (~7.4) for protonation.

    Args:
        mol: RDKit Mol object (2D)
        n_conformers: Number of conformers to generate
        random_seed: Random seed for reproducibility

    Returns:
        Mol with 3D coordinates, or None if embedding fails
    """
    try:
        # Add hydrogens (appropriate for ~pH 7.4)
        mol = Chem.AddHs(mol)

        # Generate 3D conformer using ETKDG
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        params.numThreads = 1  # Single thread per molecule for parallel safety

        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)

        if len(conf_ids) == 0:
            logger.warning("Failed to generate 3D conformer")
            return None

        # Optimize with MMFF94 force field
        for conf_id in conf_ids:
            try:
                AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)
            except Exception as e:
                logger.debug(f"MMFF optimization failed for conformer {conf_id}: {e}")
                # Try UFF as fallback
                try:
                    AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)
                except Exception:
                    pass  # Keep unoptimized conformer

        return mol

    except Exception as e:
        logger.warning(f"Error embedding molecule: {e}")
        return None


def mol_to_pdbqt(mol: Chem.Mol, output_path: Path) -> bool:
    """
    Convert RDKit Mol to PDBQT format using Meeko.

    Args:
        mol: RDKit Mol with 3D coordinates
        output_path: Where to write the PDBQT file

    Returns:
        True if successful, False otherwise
    """
    try:
        preparator = MoleculePreparation()
        mol_setup_list = preparator.prepare(mol)

        if not mol_setup_list:
            logger.warning("Meeko preparation returned empty result")
            return False

        # Get the first (and usually only) setup
        mol_setup = mol_setup_list[0]

        # Write PDBQT using PDBQTWriterLegacy
        writer = PDBQTWriterLegacy()
        pdbqt_string, is_ok, error_msg = writer.write_string(mol_setup)

        if not is_ok:
            logger.warning(f"Meeko PDBQT writing failed: {error_msg}")
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(pdbqt_string)

        return True

    except Exception as e:
        logger.warning(f"Error converting mol to PDBQT: {e}")
        return False


def prepare_ligand(
    smiles: str,
    output_dir: Path,
    index: int = 0,
    ph: Optional[float] = DEFAULT_PH,
) -> Optional[Path]:
    """
    Full ligand preparation pipeline: SMILES → protonate → 3D → PDBQT.

    Args:
        smiles: Input SMILES string
        output_dir: Directory for output files
        index: Index for naming the output file
        ph: Target pH for protonation (default 7.4, None to skip protonation)

    Returns:
        Path to PDBQT file, or None if any step fails
    """
    # Protonate for target pH (if specified)
    if ph is not None:
        smiles = protonate_smiles(smiles, ph=ph)

    # Parse SMILES
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    # Generate 3D conformer
    mol_3d = embed_molecule(mol)
    if mol_3d is None:
        return None

    # Convert to PDBQT
    output_path = output_dir / "ligands" / f"mol_{index}.pdbqt"
    if not mol_to_pdbqt(mol_3d, output_path):
        return None

    return output_path


def mol_to_sdf(mol: Chem.Mol, output_path: Path) -> bool:
    """
    Write molecule to SDF file.

    Args:
        mol: RDKit Mol with 3D coordinates
        output_path: Where to write the SDF file

    Returns:
        True if successful, False otherwise
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(output_path))
        writer.write(mol)
        writer.close()
        return True
    except Exception as e:
        logger.warning(f"Error writing SDF file: {e}")
        return False


# AutoDock/Uni-Dock supported atom types
# See: https://autodock.scripps.edu/resources/preparing-ligand-files-for-autodock/
AUTODOCK_ATOM_TYPES = frozenset({
    "H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I",
    "A", "G",  # Aromatic carbon/nitrogen
    "NA", "NS", "OA", "OS", "SA",  # H-bond acceptor variants
    "HD",  # H-bond donor hydrogen
    "Mg", "Mn", "Zn", "Ca", "Fe",  # Metals
})

# Maximum rotatable bonds for Uni-Dock (default limit is 32, but can vary)
MAX_TORSIONS_UNIDOCK = 32


def validate_pdbqt_for_unidock(pdbqt_path: Path) -> tuple[bool, str]:
    """
    Validate a PDBQT file for Uni-Dock compatibility.

    Checks for:
    - Unsupported atom types (like Boron)
    - Too many rotatable bonds/torsions

    Args:
        pdbqt_path: Path to PDBQT file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(pdbqt_path) as f:
            content = f.read()

        # Count torsions (BRANCH/ENDBRANCH pairs indicate rotatable bonds)
        n_torsions = content.count("BRANCH")

        if n_torsions > MAX_TORSIONS_UNIDOCK:
            return False, f"Too many torsions ({n_torsions} > {MAX_TORSIONS_UNIDOCK})"

        # Check atom types
        for line in content.split("\n"):
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # PDBQT format: atom type is in columns 77-78 (0-indexed: 76:78)
                # But sometimes it's at the end after charges
                parts = line.split()
                if len(parts) >= 3:
                    # Atom type is typically the last field
                    atom_type = parts[-1]
                    if atom_type not in AUTODOCK_ATOM_TYPES:
                        return False, f"Unsupported atom type: {atom_type}"

        return True, ""

    except Exception as e:
        return False, f"Error reading PDBQT: {e}"


def sanitize_smiles_for_filename(smiles: str, max_length: int = 50) -> str:
    """
    Convert SMILES string to a safe filename component.

    Args:
        smiles: SMILES string
        max_length: Maximum length of the output

    Returns:
        Sanitized string safe for use in filenames
    """
    # Replace problematic characters
    replacements = {
        "/": "_slash_",
        "\\": "_backslash_",
        "#": "_hash_",
        ":": "_colon_",
        "*": "_star_",
        "?": "_q_",
        '"': "_dq_",
        "<": "_lt_",
        ">": "_gt_",
        "|": "_pipe_",
        " ": "_",
        "(": "_lp_",
        ")": "_rp_",
        "[": "_lb_",
        "]": "_rb_",
        "=": "_eq_",
        "+": "_plus_",
        "@": "_at_",
    }

    result = smiles
    for char, replacement in replacements.items():
        result = result.replace(char, replacement)

    # Truncate if too long
    if len(result) > max_length:
        result = result[:max_length]

    return result
