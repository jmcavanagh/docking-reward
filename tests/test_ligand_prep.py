"""Tests for ligand preparation functions."""

import pytest
from rdkit import Chem

from docking_reward.ligand_prep import (
    smiles_to_mol,
    embed_molecule,
    sanitize_smiles_for_filename,
)


class TestSmilesToMol:
    """Tests for SMILES parsing."""

    def test_valid_smiles(self):
        mol = smiles_to_mol("CCO")
        assert mol is not None
        assert mol.GetNumAtoms() == 3

    def test_invalid_smiles_returns_none(self):
        mol = smiles_to_mol("invalid_smiles_xyz")
        assert mol is None

    def test_empty_smiles_returns_none(self):
        mol = smiles_to_mol("")
        assert mol is None

    def test_complex_smiles(self):
        # Aspirin
        mol = smiles_to_mol("CC(=O)Oc1ccccc1C(=O)O")
        assert mol is not None
        assert mol.GetNumAtoms() == 13


class TestEmbedMolecule:
    """Tests for 3D embedding."""

    def test_embed_simple_molecule(self):
        mol = smiles_to_mol("CCO")
        mol_3d = embed_molecule(mol)
        assert mol_3d is not None
        # Check that conformer was added
        assert mol_3d.GetNumConformers() > 0

    def test_embed_adds_hydrogens(self):
        mol = smiles_to_mol("C")  # Methane
        mol_3d = embed_molecule(mol)
        assert mol_3d is not None
        # Methane should have 5 atoms (1 C + 4 H)
        assert mol_3d.GetNumAtoms() == 5


class TestSanitizeSmiles:
    """Tests for filename sanitization."""

    def test_simple_smiles(self):
        result = sanitize_smiles_for_filename("CCO")
        assert result == "CCO"

    def test_special_characters(self):
        result = sanitize_smiles_for_filename("C/C=C\\C")
        assert "/" not in result
        assert "\\" not in result

    def test_truncation(self):
        long_smiles = "C" * 100
        result = sanitize_smiles_for_filename(long_smiles, max_length=50)
        assert len(result) == 50
