"""
Docking Reward Calculator

A tool for scoring molecules based on docking affinity, protein-ligand
interactions, and drug-likeness metrics.

Example usage:
    from docking_reward import RewardCalculator

    calc = RewardCalculator("config.yaml", n_workers=8)
    scores = calc.score(["CCO", "c1ccccc1"])

    # Or score from a file
    calc.score_file("molecules.txt", output_dir="./results")
"""

from .calculator import RewardCalculator
from .config import (
    Config,
    CustomScorerConfig,
    DruglikenessConfig,
    InteractionConfig,
    QEDConfig,
    TargetConfig,
    VinaConfig,
    load_config,
)

__version__ = "0.1.0"
__all__ = [
    "RewardCalculator",
    "Config",
    "VinaConfig",
    "TargetConfig",
    "InteractionConfig",
    "DruglikenessConfig",
    "QEDConfig",
    "CustomScorerConfig",
    "load_config",
]
