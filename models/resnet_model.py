"""
Module Description: ResNet50 Backbone Model Loader
Purpose: Loads the pre-trained ResNet50 backbone with ImageNet weights and manages layer freezing.
Author: Technical Lead
Version: 1.0.0
Dependencies: tensorflow, config.settings, utils.logger
"""

import tensorflow as tf
from config import settings
from utils.logger import app_logger


def get_resnet_backbone(freeze_backbone: bool = True) -> tf.keras.Model:
    """
    Instantiates ResNet50 backbone with ImageNet weights, excluding top classification layers.

    Args:
        freeze_backbone: If True, freezes all backbone layers.

    Returns:
        tf.keras.Model of ResNet50 backbone.
    """
    app_logger.info(f"Loading ResNet50 backbone (ImageNet weights, top=False, freeze={freeze_backbone})...")
    
    try:
        # Input shape 224x224x3 matching ImageNet specifications
        backbone = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(*settings.IMAGE_SIZE, 3)
        )
        
        # Freezing logic
        if freeze_backbone:
            backbone.trainable = False
            for layer in backbone.layers:
                layer.trainable = False
            app_logger.info("ResNet50 backbone frozen successfully.")
        else:
            backbone.trainable = True
            app_logger.info("ResNet50 backbone set as trainable.")

        return backbone
    except Exception as e:
        app_logger.error(f"Failed to load ResNet50 backbone: {e}")
        raise e
