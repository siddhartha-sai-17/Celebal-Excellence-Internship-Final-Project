"""
Module Description: Configuration Unit Tests
Purpose: Assures settings constants load with appropriate data types and formats.
Author: Technical Lead
Version: 1.0.0
Dependencies: pytest, config.settings
"""

import pytest
from config import settings


def test_project_metadata() -> None:
    """Validates metadata values."""
    assert isinstance(settings.PROJECT_NAME, str)
    assert "Fashion" in settings.PROJECT_NAME
    assert isinstance(settings.PROJECT_VERSION, str)


def test_directory_configurations() -> None:
    """Validates directory paths."""
    assert settings.BASE_DIR.exists()
    assert settings.DATASET_DIRECTORY.parent == settings.BASE_DIR
    assert settings.SUBSET_DIRECTORY.exists()
    assert settings.LOG_DIRECTORY.exists()


def test_preprocessing_settings() -> None:
    """Validates image sizing and formats."""
    assert settings.IMAGE_SIZE == (224, 224)
    assert ".jpg" in settings.SUPPORTED_IMAGE_FORMATS
    assert len(settings.IMAGENET_MEAN) == 3
    assert len(settings.IMAGENET_STD) == 3


def test_hyperparameters() -> None:
    """Validates optimization hyperparameters."""
    assert settings.EMBEDDING_DIMENSION > 0
    assert settings.BATCH_SIZE > 0
    assert settings.LEARNING_RATE > 0
    assert settings.TOP_K_RESULTS > 0
