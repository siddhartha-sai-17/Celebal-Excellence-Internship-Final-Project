"""
Module Description: Siamese Network Inference
Purpose: Predicts similarity distances between query images and targets using the Siamese model.
Author: Technical Lead
Version: 1.0.0
Dependencies: tensorflow, PIL.Image, config.settings, utils.logger, preprocessing.image_preprocessor
"""

from pathlib import Path
from typing import Union, List, Tuple
import numpy as np
import tensorflow as tf
from PIL import Image
from config import settings
from utils.logger import inference_logger
from preprocessing.image_preprocessor import ImagePreprocessor


class SiameseInference:
    """
    Performs distance prediction and similarity calculations using a loaded Siamese pair model.
    """

    def __init__(self, siamese_model: tf.keras.Model) -> None:
        """
        Initializes the inference engine with a loaded pair model.

        Args:
            siamese_model: The Keras functional Siamese Pair Model.
        """
        self.model: tf.keras.Model = siamese_model

    def predict_distance(self, 
                         image1: Union[str, Path, Image.Image], 
                         image2: Union[str, Path, Image.Image]) -> float:
        """
        Computes the Euclidean distance between two images.

        Args:
            image1: First image.
            image2: Second image.

        Returns:
            Euclidean distance value.
        """
        tensor1 = ImagePreprocessor.preprocess_for_inference(image1)
        tensor2 = ImagePreprocessor.preprocess_for_inference(image2)

        if tensor1 is None or tensor2 is None:
            inference_logger.error("Failed to preprocess images for distance prediction.")
            return 999.0

        try:
            # Run model prediction
            distance = self.model([tensor1, tensor2], training=False)
            val = float(distance.numpy().squeeze())
            inference_logger.info(f"Predicted pair distance: {val:.4f}")
            return val
        except Exception as e:
            inference_logger.error(f"Failed to run Siamese distance prediction: {e}")
            return 999.0
