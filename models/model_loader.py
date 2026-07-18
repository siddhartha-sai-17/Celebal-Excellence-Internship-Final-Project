"""
Module Description: Safe Model Loader
Purpose: Safely instantiates and restores model architectures and weights from files.
Author: Technical Lead
Version: 1.0.0
Dependencies: tensorflow, pathlib, config.settings, utils.logger, models.model_factory, models.checkpoint_manager
"""

from pathlib import Path
from typing import Optional
import tensorflow as tf
from config import settings
from utils.logger import app_logger
from models.model_factory import ModelFactory
from models.checkpoint_manager import CheckpointManager


class ModelLoader:
    """
    Helper class to load fully serialized models or restore trained weights into fresh architectures.
    """

    @staticmethod
    def load_weights_into_model(model: tf.keras.Model, weights_path: Path) -> bool:
        """
        Loads weights into an existing instantiated model.

        Args:
            model: Instantiated Keras Model.
            weights_path: Path to weights file.

        Returns:
            True if weights were loaded successfully, False otherwise.
        """
        if not weights_path.exists():
            app_logger.error(f"Weights file not found: {weights_path}")
            return False

        app_logger.info(f"Loading weights into model from {weights_path}...")
        try:
            model.load_weights(str(weights_path))
            app_logger.info("Weights loaded successfully.")
            return True
        except Exception as e:
            app_logger.error(f"Failed to load weights: {e}")
            return False

    @staticmethod
    def load_embedding_model(model_name: str = settings.MODEL_NAME, 
                             checkpoint_dir: Optional[Path] = None) -> Optional[tf.keras.Model]:
        """
        Builds the embedding model and, if checkpoint_dir is provided, loads the weights.

        Args:
            model_name: Name of backbone (e.g. 'resnet50').
            checkpoint_dir: Directory containing model weights or model.keras.

        Returns:
            tf.keras.Model containing the embedding model, or None if building/loading failed.
        """
        try:
            # Rebuild architecture
            model = ModelFactory.build_embedding_model(
                model_name=model_name,
                embedding_dim=settings.EMBEDDING_DIMENSION,
                dropout_rate=settings.DROPOUT_RATE,
                freeze_backbone=False  # Make it false so weights can load trainable values if any
            )

            if checkpoint_dir is not None:
                keras_file = checkpoint_dir / "model.keras"
                if keras_file.exists():
                    app_logger.info(f"Loading full model from {keras_file}...")
                    try:
                        # Load custom objects for Custom Layer deserialization
                        from models.embedding_head import EmbeddingHead
                        loaded_model = tf.keras.models.load_model(
                            keras_file,
                            custom_objects={"EmbeddingHead": EmbeddingHead}
                        )
                        # We must return the loaded model
                        return loaded_model
                    except Exception as e:
                        app_logger.error(f"Failed to load full Keras model: {e}. Falling back to weights load...")

                # Fallback to loading weights if weights are separate
                weights_file = checkpoint_dir / "weights.h5"
                if weights_file.exists():
                    success = ModelLoader.load_weights_into_model(model, weights_file)
                    if not success:
                        app_logger.warning("Could not restore weights from H5 file.")

            return model
        except Exception as e:
            app_logger.error(f"Error initializing embedding model: {e}")
            return None
