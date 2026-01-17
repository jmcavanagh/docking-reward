"""Command-line interface for docking reward calculator."""

import argparse
import logging
import sys
from pathlib import Path

from .calculator import RewardCalculator


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    """
    CLI entry point: docking-reward

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = argparse.ArgumentParser(
        prog="docking-reward",
        description="Score molecules based on docking affinity, interactions, and drug-likeness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  docking-reward -i molecules.txt -o ./results -c config.yaml

  # With parallel processing
  docking-reward -i molecules.txt -o ./results -c config.yaml -w 16

  # Using Dask backend
  docking-reward -i molecules.txt -o ./results -c config.yaml -w 64 -b dask

Output:
  Creates output_dir with:
    - results.csv: Two columns (smiles, score)
    - poses/: Directory with SDF files for each valid molecule
    - tmp/: Intermediate PDBQT files (kept for debugging)
        """,
    )

    parser.add_argument(
        "-i", "--input-smiles",
        type=Path,
        required=True,
        help="Path to file with SMILES strings (one per line)",
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        required=True,
        help="Directory for results and poses",
    )

    parser.add_argument(
        "-c", "--config-file",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )

    parser.add_argument(
        "-b", "--backend",
        choices=["multiprocessing", "dask"],
        default="multiprocessing",
        help="Parallelization backend (default: multiprocessing)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate inputs
    if not args.input_smiles.exists():
        logger.error(f"Input file not found: {args.input_smiles}")
        return 2

    if not args.config_file.exists():
        logger.error(f"Config file not found: {args.config_file}")
        return 1

    # Set temp directory inside output directory
    temp_dir = args.output_dir / "tmp"

    try:
        # Create calculator
        logger.info(f"Loading configuration from {args.config_file}")
        calc = RewardCalculator(
            config_path=args.config_file,
            n_workers=args.workers,
            backend=args.backend,
            temp_dir=temp_dir,
        )

        # Run scoring
        logger.info(f"Processing SMILES from {args.input_smiles}")
        results_path = calc.score_file(
            input_path=args.input_smiles,
            output_dir=args.output_dir,
        )

        logger.info(f"Results written to {results_path}")

        # Clean up
        calc.shutdown()

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 2

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1

    except Exception as e:
        logger.exception(f"Runtime error: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
