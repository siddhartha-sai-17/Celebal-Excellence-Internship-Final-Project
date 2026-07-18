"""
Module Description: EfficientNetB0 Backbone Model Loader
Purpose: Loads the pre-trained EfficientNetB0 backbone with ImageNet weights and manages layer freezing.
Author: Technical Lead
Version: 1.0.0
Dependencies: tensorflow, config.settings, utils.logger
"""

import tensorflow as tf
from config import settings
from utils.logger import app_logger


def get_efficientnet_backbone(freeze_backbone: bool = True) -> tf.keras.Model:
    """
    Instantiates EfficientNetB0 backbone with ImageNet weights, excluding top classification layers.

    Args:
        freeze_backbone: If True, freezes all backbone layers.

    Returns:
        tf.keras.Model of EfficientNetB0 backbone.
    """
    app_logger.info(f"Loading EfficientNetB0 backbone (ImageNet weights, top=False, freeze={freeze_backbone})...")
    
    try:
        # Input shape 224x224x3 matching ImageNet specifications
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(*settings.IMAGE_SIZE, 3)
        )
        
        # Freezing logic
        if freeze_backbone:
            backbone.trainable = False
            for layer in backbone.layers:
                layer.trainable = False
            app_logger.info("EfficientNetB0 backbone frozen successfully.")
        else:
            backbone.trainable = True
            app_logger.info("EfficientNetB0 backbone set as trainable.")

        return backbone
    except Exception as e:
        app_logger.error(f"Failed to load EfficientNetB0 backbone: {e}")
        raise e
