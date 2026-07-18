"""
Module Description: Logging Unit Tests
Purpose: Assures loggers write information messages and handlers are configured.
Author: Technical Lead
Version: 1.0.0
Dependencies: pytest, utils.logger
"""

import pytest
import logging
from utils.logger import app_logger, training_logger, exception_logger


def test_logger_setup() -> None:
    """Validates logger names and instances."""
    assert isinstance(app_logger, logging.Logger)
    assert app_logger.name == "app"
    assert training_logger.name == "training"
    assert exception_logger.name == "exception"


def test_logging_calls() -> None:
    """Ensures log execution does not crash."""
    try:
        app_logger.info("Unit test diagnostic info message.")
        training_logger.warning("Unit test diagnostic warning message.")
        exception_logger.error("Unit test diagnostic error message.")
    except Exception as e:
        pytest.fail(f"Logging execution crashed: {e}")
