#!/usr/bin/env python3
"""Unified logging setup for training and serving."""

import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Setup consistent logging across the pipeline.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("ml_pipeline")
    logger.setLevel(getattr(logging, level.upper()))

    # Only add handlers if none exist to prevent duplicates
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
