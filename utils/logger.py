"""
Module Description: Centralized Logging Module
Purpose: Sets up separate, dedicated loggers and log files for application, training, recommendation, inference, performance, and exceptions.
Author: Technical Lead
Version: 1.0.0
Dependencies: logging, config.settings
"""

import logging
import sys
from pathlib import Path
from typing import Dict
from config import settings


def setup_logger(name: str, log_filename: str, level: str = settings.LOG_LEVEL) -> logging.Logger:
    """
    Configures and returns a logger with both a console handler and a file handler.

    Args:
        name: Name of the logger.
        log_filename: Name of the file where logs will be stored.
        level: Logging level (e.g., "INFO", "DEBUG", "ERROR").

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent adding duplicate handlers if the logger was already configured
    if logger.handlers:
        return logger

    # Ensure logs folder exists
    log_dir = Path(settings.LOG_DIRECTORY)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filepath = log_dir / log_filename

    # Create formatters
    formatter = logging.Formatter(settings.LOG_FORMAT)

    # Console Handler (writes to standard output with UTF-8 to handle emoji safely on Windows)
    console_handler = logging.StreamHandler(sys.stdout)
    # Force UTF-8 encoding on Windows where default is cp1252
    if hasattr(console_handler.stream, 'reconfigure'):
        try:
            console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback console log for warning of permission issues
        print(f"Warning: Could not create log file handler for {log_filepath} due to: {e}", file=sys.stderr)

    return logger


# Pre-configured loggers for Clean Architecture layers
app_logger: logging.Logger = setup_logger("app", "app.log")
training_logger: logging.Logger = setup_logger("training", "training.log")
recommendation_logger: logging.Logger = setup_logger("recommendation", "recommendation.log")
inference_logger: logging.Logger = setup_logger("inference", "inference.log")
performance_logger: logging.Logger = setup_logger("performance", "performance.log")
exception_logger: logging.Logger = setup_logger("exception", "exception.log", level="ERROR")
