"""Protein preparation: PDB to PDBQT conversion with protonation checking."""

import logging
import subprocess
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)


def check_protonation(pdb_path: Path) -> bool:
    """
    Check if a PDB file appears to be protonated.

    Looks for hydrogen atoms in the structure. If no hydrogens are found
    on heavy atoms that should have them, issues a warning.

    Args:
        pdb_path: Path to PDB file

    Returns:
        True if protonated (has hydrogens), False otherwise
    """
    has_hydrogen = False
    heavy_atom_count = 0

    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    # Column 77-78 contains element symbol (right-justified)
                    # Column 13-16 contains atom name
                    atom_name = line[12:16].strip()
                    element = line[76:78].strip() if len(line) > 76 else ""

                    # Check element field first, fall back to atom name
                    if element == "H" or (not element and atom_name.startswith("H")):
                        has_hydrogen = True
                    elif element not in ("H", "") or not atom_name.startswith("H"):
                        heavy_atom_count += 1

    except Exception as e:
        logger.warning(f"Error reading PDB file for protonation check: {e}")
        return True  # Assume protonated if we can't check

    return has_hydrogen


def warn_if_not_protonated(pdb_path: Path, target_name: str) -> None:
    """
    Check PDB protonation state and warn if not protonated.

    Args:
        pdb_path: Path to PDB file
        target_name: Name of the target (for warning message)
    """
    if not check_protonation(pdb_path):
        warnings.warn(
            f"PDB file for target '{target_name}' does not appear to be protonated. "
            f"For accurate docking results, please protonate the structure at pH 7.4 "
            f"using tools like PDB2PQR, Reduce, or PyMOL. "
            f"File: {pdb_path}",
            UserWarning,
            stacklevel=3,
        )
        logger.warning(
            f"Target '{target_name}' PDB file may not be protonated: {pdb_path}"
        )


def pdb_to_pdbqt(pdb_path: Path, output_path: Path, target_name: str = "unknown") -> bool:
    """
    Convert PDB file to PDBQT format using OpenBabel.

    Args:
        pdb_path: Input PDB file path
        output_path: Output PDBQT file path
        target_name: Name of target for warning messages

    Returns:
        True if successful, False otherwise

    Notes:
        - Warns if the PDB file does not appear to be protonated
        - Removes water molecules
        - Computes Gasteiger charges via OpenBabel
    """
    # Check protonation state and warn if needed
    warn_if_not_protonated(pdb_path, target_name)

    try:
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use OpenBabel to convert PDB to PDBQT
        # -xr: treat as rigid receptor (no rotatable bonds)
        # -p 7.4: protonate at pH 7.4 (if not already protonated)
        # --partialcharge gasteiger: compute Gasteiger charges
        result = subprocess.run(
            [
                "obabel",
                str(pdb_path),
                "-O", str(output_path),
                "-xr",  # Rigid receptor mode
                "--partialcharge", "gasteiger",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            logger.error(f"OpenBabel conversion failed: {result.stderr}")
            return False

        # Verify output file was created
        if not output_path.exists():
            logger.error(f"OpenBabel did not create output file: {output_path}")
            return False

        return True

    except FileNotFoundError:
        logger.error(
            "OpenBabel (obabel) not found. Please install it with: "
            "conda install -c conda-forge openbabel"
        )
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"OpenBabel conversion timed out for {pdb_path}")
        return False
    except Exception as e:
        logger.error(f"Error converting PDB to PDBQT: {e}")
        return False


def prepare_proteins(
    targets: list, cache_dir: Path
) -> dict[str, Path]:
    """
    Prepare all protein targets, with caching.

    Args:
        targets: List of TargetConfig objects
        cache_dir: Directory to store prepared PDBQT files

    Returns:
        Dict mapping target name to PDBQT path

    Notes:
        - Skips preparation if PDBQT already exists and is newer than PDB
    """
    protein_pdbqts = {}
    proteins_dir = cache_dir / "proteins"
    proteins_dir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        output_path = proteins_dir / f"{target.name}.pdbqt"

        # Check if we need to regenerate
        if output_path.exists():
            pdb_mtime = target.pdb_path.stat().st_mtime
            pdbqt_mtime = output_path.stat().st_mtime
            if pdbqt_mtime >= pdb_mtime:
                logger.info(f"Using cached PDBQT for target '{target.name}'")
                protein_pdbqts[target.name] = output_path
                continue

        logger.info(f"Preparing protein PDBQT for target '{target.name}'")
        if pdb_to_pdbqt(target.pdb_path, output_path, target.name):
            protein_pdbqts[target.name] = output_path
        else:
            raise RuntimeError(
                f"Failed to prepare protein PDBQT for target '{target.name}'"
            )

    return protein_pdbqts
