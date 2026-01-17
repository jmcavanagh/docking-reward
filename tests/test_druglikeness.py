"""Tests for drug-likeness scoring."""

import pytest
from rdkit import Chem

from docking_reward.druglikeness import calculate_qed, DruglikenessScorer
from docking_reward.config import DruglikenessConfig, QEDConfig


class TestCalculateQED:
    """Tests for QED calculation."""

    def test_qed_returns_float(self):
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        qed = calculate_qed(mol)
        assert isinstance(qed, float)

    def test_qed_in_range(self):
        # QED should always be between 0 and 1
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
        mol = Chem.AddHs(mol)
        qed = calculate_qed(mol)
        assert 0.0 <= qed <= 1.0

    def test_qed_drug_like_molecule(self):
        # Aspirin should have a reasonable QED
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        mol = Chem.AddHs(mol)
        qed = calculate_qed(mol)
        assert qed > 0.3  # Aspirin is reasonably drug-like


class TestDruglikenessScorer:
    """Tests for the DruglikenessScorer class."""

    def test_no_scorers_returns_zero(self):
        config = DruglikenessConfig()
        scorer = DruglikenessScorer(config)
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        score = scorer.score(mol)
        assert score == 0.0

    def test_qed_weighted_score(self):
        config = DruglikenessConfig(qed=QEDConfig(weight=2.0))
        scorer = DruglikenessScorer(config)
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        score = scorer.score(mol)
        # Score should be weight * qed
        expected_qed = calculate_qed(mol)
        assert abs(score - 2.0 * expected_qed) < 0.001
