"""Command-line interface for docking reward calculator."""

import argparse
import logging
import sys
from pathlib import Path

from .calculator import RewardCalculator
from .logging_config import setup_logging


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
  # Basic usage (CPU Vina backend, set in config.yaml)
  docking-reward -i molecules.txt -o ./results -c config.yaml

  # With parallel processing (64 CPU workers)
  docking-reward -i molecules.txt -o ./results -c config.yaml -w 64

  # Using Dask for distributed computing
  docking-reward -i molecules.txt -o ./results -c config.yaml -w 64 -b dask

Docking backends (set in config.yaml):
  - vina: CPU-based AutoDock Vina (parallelized per molecule)
  - unidock: GPU-based Uni-Dock (batched, much faster with GPU)

Scoring functions (set in config.yaml):
  - vina: Standard Vina scoring
  - vinardo: Faster Vinardo scoring
  - ad4: AutoDock4 scoring

Output:
  Creates output_dir with:
    - results.csv: Two columns (smiles, score)
    - breakdown.csv: Detailed score breakdown
    - docking.log: Detailed log with all warnings (RDKit, Vina, etc.)
    - poses/: Directory with SDF files for each valid molecule
    - tmp/: Intermediate PDBQT files (kept for debugging)

  Console shows clean progress bars and summary stats.
  All verbose warnings from RDKit/Vina go to docking.log.
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
        "-b", "--parallel-backend",
        choices=["multiprocessing", "dask"],
        default="multiprocessing",
        help="CPU parallelization backend (default: multiprocessing). "
             "Note: Docking backend (vina/unidock) is set in config file.",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show debug messages on console",
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress console output (only show errors)",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path for detailed log file (default: OUTPUT_DIR/docking.log). "
             "All RDKit/Vina warnings go here instead of console.",
    )

    args = parser.parse_args()

    # Set log file path (default to output_dir/docking.log)
    log_file = args.log_file
    if log_file is None:
        log_file = args.output_dir / "docking.log"

    # Set up logging with clean console output and detailed log file
    setup_logging(
        log_file=log_file,
        verbose=args.verbose,
        quiet=args.quiet,
    )
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
            parallel_backend=args.parallel_backend,
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
