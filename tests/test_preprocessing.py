"""
Module Description: Preprocessing Unit Tests
Purpose: Assures PIL loading, sizing, normalization, and augmentation layers perform correctly.
Author: Technical Lead
Version: 1.0.0
Dependencies: pytest, PIL.Image, numpy, tensorflow, preprocessing.image_validator, preprocessing.image_loader, preprocessing.image_preprocessor, preprocessing.augmentation
"""

import os
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import tensorflow as tf
from config import settings
from preprocessing.image_validator import ImageValidator
from preprocessing.image_loader import ImageLoader
from preprocessing.image_preprocessor import ImagePreprocessor
from preprocessing.augmentation import DataAugmentor


@pytest.fixture
def dummy_image_path(tmp_path: Path) -> Path:
    """Creates a temporary dummy JPG image for testing."""
    img = Image.new("RGB", (100, 100), color="red")
    path = tmp_path / "test_img.jpg"
    img.save(path)
    return path


@pytest.fixture
def dummy_corrupt_path(tmp_path: Path) -> Path:
    """Creates a corrupted image file."""
    path = tmp_path / "corrupt_img.jpg"
    with open(path, "wb") as f:
        f.write(b"corrupt image content header bytes")
    return path


def test_image_validator(dummy_image_path: Path, dummy_corrupt_path: Path) -> None:
    """Validates image checker outcomes."""
    # Valid image
    is_valid, msg = ImageValidator.validate_image_file(dummy_image_path)
    assert is_valid
    assert msg == ""

    # Non-existent image
    is_valid, msg = ImageValidator.validate_image_file("nonexistent.jpg")
    assert not is_valid
    assert "exist" in msg

    # Corrupt image
    is_valid, msg = ImageValidator.validate_image_file(dummy_corrupt_path)
    assert not is_valid
    assert "Corrupted" in msg


def test_image_loader(dummy_image_path: Path) -> None:
    """Ensures loading yields RGB PIL images."""
    img = ImageLoader.load_image(dummy_image_path)
    assert img is not None
    assert img.mode == "RGB"
    assert img.size == (100, 100)


def test_image_preprocessor(dummy_image_path: Path) -> None:
    """Ensures preprocessing resizes to 224x224 and normalizes."""
    tensor = ImagePreprocessor.preprocess_image(dummy_image_path)
    assert tensor is not None
    assert isinstance(tensor, tf.Tensor)
    assert tensor.shape == (224, 224, 3)

    # Batch dimension preprocessing
    batch_tensor = ImagePreprocessor.preprocess_for_inference(dummy_image_path)
    assert batch_tensor is not None
    assert batch_tensor.shape == (1, 224, 224, 3)


def test_data_augmentor() -> None:
    """Ensures data augmentation layer runs without exceptions."""
    augmentor = DataAugmentor(enabled=True)
    layer = augmentor.get_augmentation_layer()
    assert isinstance(layer, tf.keras.layers.Layer)

    # Dummy forward pass (batch size 1)
    dummy_input = tf.random.normal((1, 224, 224, 3))
    output = layer(dummy_input, training=True)
    assert output.shape == (1, 224, 224, 3)
