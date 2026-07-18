"""
Module Description: Data Augmentation Layer Pipeline
Purpose: Implements horizontal flips, random rotation, zoom, brightness, contrast, and translation for training data.
Author: Technical Lead
Version: 1.0.0
Dependencies: tensorflow, config.settings, utils.logger
"""

import tensorflow as tf
from config import settings
from utils.logger import app_logger


class DataAugmentor:
    """
    Wrapper class to define and apply data augmentation layers in Keras.
    """

    def __init__(self, enabled: bool = True) -> None:
        """
        Initializes the data augmentor.

        Args:
            enabled: Boolean indicating whether to apply augmentation layers.
        """
        self.enabled: bool = enabled
        self.pipeline: tf.keras.Sequential = tf.keras.Sequential()

        if self.enabled:
            # Build tf.keras.layers augmentation sequence
            self.pipeline.add(tf.keras.layers.RandomFlip("horizontal", seed=settings.RANDOM_SEED))
            self.pipeline.add(tf.keras.layers.RandomRotation(0.15, fill_mode="reflect", seed=settings.RANDOM_SEED))
            self.pipeline.add(tf.keras.layers.RandomZoom(0.1, 0.1, fill_mode="reflect", seed=settings.RANDOM_SEED))
            self.pipeline.add(tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", seed=settings.RANDOM_SEED))
            
            # Brightness and Contrast layers (RandomBrightness, RandomContrast)
            # Use factor or range bounds appropriately
            self.pipeline.add(tf.keras.layers.RandomBrightness(factor=0.1, value_range=(0.0, 1.0), seed=settings.RANDOM_SEED))
            self.pipeline.add(tf.keras.layers.RandomContrast(factor=0.1, seed=settings.RANDOM_SEED))
            
            app_logger.info("Data augmentation pipeline initialized with layers: Flip, Rotation, Zoom, Translation, Brightness, Contrast.")
        else:
            app_logger.info("Data augmentation pipeline is disabled.")

    def get_augmentation_layer(self) -> tf.keras.layers.Layer:
        """
        Returns the data augmentation pipeline as a single Keras Layer.

        Returns:
            tf.keras.Sequential or Identity layer.
        """
        if self.enabled:
            return self.pipeline
        return tf.keras.layers.Activation("linear")
