"""
Module Description: Image Preprocessing Pipeline
Purpose: Handles resizing to 224x224, ImageNet normalization, and conversion to TensorFlow tensors.
Author: Technical Lead
Version: 1.0.0
Dependencies: numpy, tensorflow, PIL.Image, config.settings, utils.logger, preprocessing.image_loader
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import tensorflow as tf
from PIL import Image
from config import settings
from utils.logger import app_logger
from preprocessing.image_loader import ImageLoader


class ImagePreprocessor:
    """
    Provides static methods to preprocess single images or batch of images.
    """

    @staticmethod
    def preprocess_image(image_input: Union[str, Path, Image.Image]) -> Optional[tf.Tensor]:
        """
        Reads an image (if path is provided), resizes it to target size, normalizes
        using ImageNet statistics, and returns it as a float32 tensor of shape (224, 224, 3).

        Args:
            image_input: Path to image or PIL Image object.

        Returns:
            Preprocessed TensorFlow tensor of shape (224, 224, 3) or None if loading/processing failed.
        """
        img: Optional[Image.Image] = None

        if isinstance(image_input, (str, Path)):
            img = ImageLoader.load_image(image_input)
            if img is None:
                return None
        elif isinstance(image_input, Image.Image):
            # Ensure RGB
            if image_input.mode != "RGB":
                img = image_input.convert("RGB")
            else:
                img = image_input
        else:
            app_logger.error(f"Invalid image input type: {type(image_input)}")
            return None

        try:
            # Resize image to target dimensions (224x224) using bilinear interpolation
            img_resized = img.resize(settings.IMAGE_SIZE, Image.Resampling.BILINEAR)

            # Convert to numpy array and scale to [0, 1] range
            img_arr = np.array(img_resized, dtype=np.float32) / 255.0

            # Normalize using ImageNet statistics: (pixel - mean) / std
            mean = np.array(settings.IMAGENET_MEAN, dtype=np.float32)
            std = np.array(settings.IMAGENET_STD, dtype=np.float32)
            img_normalized = (img_arr - mean) / std

            # Convert to TensorFlow tensor
            tensor = tf.convert_to_tensor(img_normalized, dtype=tf.float32)
            return tensor
        except Exception as e:
            app_logger.error(f"Image preprocessing failed: {e}")
            return None

    @staticmethod
    def preprocess_for_inference(image_input: Union[str, Path, Image.Image]) -> Optional[tf.Tensor]:
        """
        Same as preprocess_image, but prepends batch dimension. Returns shape (1, 224, 224, 3).

        Args:
            image_input: Path to image or PIL Image object.

        Returns:
            Tensor of shape (1, 224, 224, 3) or None.
        """
        tensor = ImagePreprocessor.preprocess_image(image_input)
        if tensor is None:
            return None
        return tf.expand_dims(tensor, axis=0)
