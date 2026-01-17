"""Drug-likeness scoring: QED and custom scorer plugins."""

import importlib.util
import logging
from pathlib import Path
from typing import Callable, Optional

from rdkit import Chem
from rdkit.Chem import QED as RDKitQED

from .config import DruglikenessConfig

logger = logging.getLogger(__name__)


def calculate_qed(mol: Chem.Mol) -> float:
    """
    Calculate QED (Quantitative Estimate of Drug-likeness).

    Args:
        mol: RDKit Mol object

    Returns:
        QED score between 0 and 1 (higher = more drug-like)
    """
    try:
        # Ensure molecule has no Hs for consistent calculation
        mol_no_h = Chem.RemoveHs(mol)
        return RDKitQED.qed(mol_no_h)
    except Exception as e:
        logger.warning(f"QED calculation failed: {e}")
        return 0.0


def load_custom_scorer(path: Path, function_name: str) -> Optional[Callable[[Chem.Mol], float]]:
    """
    Dynamically load a custom scoring function from a Python file.

    The custom scorer file should define a function with signature:
        def score(mol: rdkit.Chem.Mol) -> float

    Args:
        path: Path to Python file
        function_name: Name of function to load

    Returns:
        Callable that takes an RDKit Mol and returns a float score,
        or None if loading fails

    Example custom scorer file:
        ```python
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        def score(mol: Chem.Mol) -> float:
            '''Custom scorer based on molecular weight.'''
            mw = Descriptors.MolWt(mol)
            # Prefer molecules around 400 Da
            return 1.0 - abs(mw - 400) / 400
        ```
    """
    try:
        # Load the module from file
        spec = importlib.util.spec_from_file_location("custom_scorer", path)
        if spec is None or spec.loader is None:
            logger.error(f"Could not load module spec from {path}")
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the function
        if not hasattr(module, function_name):
            logger.error(f"Function '{function_name}' not found in {path}")
            return None

        func = getattr(module, function_name)

        if not callable(func):
            logger.error(f"'{function_name}' in {path} is not callable")
            return None

        return func

    except Exception as e:
        logger.error(f"Error loading custom scorer from {path}: {e}")
        return None


class DruglikenessScorer:
    """
    Manages drug-likeness scoring with QED and custom scorers.

    This class loads and caches custom scorers and provides a unified
    interface for calculating drug-likeness contributions.
    """

    def __init__(self, config: DruglikenessConfig):
        """
        Initialize the drug-likeness scorer.

        Args:
            config: Drug-likeness configuration
        """
        self.config = config
        self._custom_scorers: list[tuple[Callable, float]] = []

        # Load custom scorers
        for custom_config in config.custom:
            scorer = load_custom_scorer(custom_config.path, custom_config.function)
            if scorer is not None:
                self._custom_scorers.append((scorer, custom_config.weight))
                logger.info(
                    f"Loaded custom scorer '{custom_config.function}' "
                    f"from {custom_config.path} with weight {custom_config.weight}"
                )
            else:
                logger.warning(
                    f"Failed to load custom scorer from {custom_config.path}"
                )

    def score(self, mol: Chem.Mol) -> float:
        """
        Calculate total drug-likeness score.

        Args:
            mol: RDKit Mol object

        Returns:
            Weighted sum of all drug-likeness scores
        """
        total_score = 0.0

        # QED contribution
        if self.config.qed is not None:
            qed_score = calculate_qed(mol)
            contribution = self.config.qed.weight * qed_score
            total_score += contribution
            logger.debug(f"QED score: {qed_score:.4f}, contribution: {contribution:.4f}")

        # Custom scorer contributions
        for scorer_func, weight in self._custom_scorers:
            try:
                custom_score = scorer_func(mol)
                contribution = weight * custom_score
                total_score += contribution
                logger.debug(f"Custom score: {custom_score:.4f}, contribution: {contribution:.4f}")
            except Exception as e:
                logger.warning(f"Custom scorer failed: {e}")

        return total_score


def calculate_druglikeness(mol: Chem.Mol, config: DruglikenessConfig) -> float:
    """
    Calculate total drug-likeness score from all configured scorers.

    This is a convenience function that creates a temporary scorer.
    For repeated calls, use DruglikenessScorer class directly.

    Args:
        mol: RDKit Mol object
        config: Drug-likeness configuration

    Returns:
        Weighted sum of all drug-likeness scores
    """
    scorer = DruglikenessScorer(config)
    return scorer.score(mol)
