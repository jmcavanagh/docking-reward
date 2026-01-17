"""Logging configuration for docking reward calculator.

Separates verbose library warnings (RDKit, Vina, etc.) from main progress output.
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

# Suppress RDKit warnings at import time
os.environ["RDK_LOG_LEVEL"] = "ERROR"


def setup_logging(
    log_file: Optional[Path] = None,
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    """
    Configure logging for the docking reward calculator.

    Args:
        log_file: Path to write detailed logs (including all warnings).
                  If None, writes to output_dir/docking.log
        verbose: If True, show debug messages on console
        quiet: If True, suppress most console output (only errors)

    The setup:
    - Console: Clean output with progress bars (INFO level by default)
    - Log file: All detailed warnings from RDKit, Vina, etc. (DEBUG level)
    """
    # Determine console log level
    if quiet:
        console_level = logging.ERROR
    elif verbose:
        console_level = logging.DEBUG
    else:
        console_level = logging.INFO

    # Root logger setup
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything

    # Clear any existing handlers
    root_logger.handlers = []

    # Console handler - clean output
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        "%(message)s"  # Simple format for console
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler - detailed output (if log_file specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Suppress noisy loggers on console (but let them go to file)
    noisy_loggers = [
        "rdkit",
        "meeko",
        "vina",
        "openbabel",
        "plip",
    ]
    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        # Don't propagate to root (we handle them separately)
        # But still allow file logging via explicit handler
        logger.setLevel(logging.WARNING)

    # Capture Python warnings and send to log file
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.setLevel(logging.WARNING)

    # Suppress specific RDKit/chemistry warnings from console
    # These will still go to the log file
    class ChemistryWarningFilter(logging.Filter):
        """Filter out chemistry library warnings from console."""

        def filter(self, record: logging.LogRecord) -> bool:
            # Allow through if this is going to file
            if any(h.level == logging.DEBUG for h in record.handlers if hasattr(h, 'level')):
                return True

            # Filter out common noisy messages
            noisy_patterns = [
                "Warning: molecule",
                "WARNING: molecule",
                "Sanitization",
                "Can't kekulize",
                "Explicit valence",
                "SMILES Parse Error",
                "rdkit",
            ]
            msg = str(record.msg)
            for pattern in noisy_patterns:
                if pattern.lower() in msg.lower():
                    return False
            return True

    console_handler.addFilter(ChemistryWarningFilter())


def suppress_rdkit_warnings() -> None:
    """Suppress RDKit warnings globally.

    Call this before importing rdkit for best effect.
    """
    # Environment variable method
    os.environ["RDK_LOG_LEVEL"] = "ERROR"

    # Try to suppress via rdkit logger if already imported
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        pass

    # Suppress Python warnings from rdkit
    warnings.filterwarnings("ignore", category=UserWarning, module="rdkit")
    warnings.filterwarnings("ignore", message=".*RDKit.*")


# Auto-suppress on import
suppress_rdkit_warnings()
