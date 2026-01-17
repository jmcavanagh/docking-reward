"""Tests for configuration parsing."""

import pytest
import tempfile
from pathlib import Path

from docking_reward.config import (
    load_config,
    VinaConfig,
    TargetConfig,
    InteractionConfig,
    DruglikenessConfig,
    QEDConfig,
    VALID_INTERACTION_TYPES,
)


class TestVinaConfig:
    """Tests for VinaConfig defaults."""

    def test_defaults(self):
        config = VinaConfig()
        assert config.exhaustiveness == 8
        assert config.n_poses == 9
        assert config.energy_range == 3.0
        assert config.seed is None


class TestInteractionTypes:
    """Tests for valid interaction types."""

    def test_valid_types(self):
        expected = {"hydrogen_bond", "hydrophobic", "salt_bridge", "pi_stacking", "any"}
        assert VALID_INTERACTION_TYPES == expected


class TestConfigLoading:
    """Tests for YAML config loading."""

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_empty_config_raises(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        with pytest.raises(ValueError, match="empty"):
            load_config(config_file)

    def test_no_targets_raises(self, tmp_path):
        config_file = tmp_path / "no_targets.yaml"
        config_file.write_text("""
vina:
  exhaustiveness: 8
druglikeness:
  qed:
    weight: 1.0
targets: []
""")
        with pytest.raises(ValueError, match="at least one target"):
            load_config(config_file)
