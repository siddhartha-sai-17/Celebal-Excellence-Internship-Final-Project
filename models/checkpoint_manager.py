"""
Module Description: Model Checkpoint Manager
Purpose: Saves and restores model checkpoints, optimizer state, training histories, and generates version JSON logs.
Author: Technical Lead
Version: 1.0.0
Dependencies: json, datetime, pathlib, tensorflow, config.settings, utils.logger
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any, Optional
import tensorflow as tf
from config import settings
from utils.logger import app_logger


class CheckpointManager:
    """
    Manages saving and loading model checkpoints and training history metadata.
    """

    def __init__(self, checkpoints_dir: Path = settings.MODEL_DIRECTORY / "checkpoints") -> None:
        """
        Initializes the manager.

        Args:
            checkpoints_dir: Folder to save checkpoints.
        """
        self.checkpoints_dir: Path = checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, 
                        model: tf.keras.Model, 
                        model_name: str, 
                        version: str, 
                        history_data: Optional[Dict[str, Any]] = None,
                        metrics_summary: Optional[Dict[str, Any]] = None) -> Path:
        """
        Saves model weights and exports a JSON metadata config.

        Args:
            model: The Keras model to save.
            model_name: Name identifier (e.g. 'resnet50_transfer').
            version: Model version tag (e.g. 'v1').
            history_data: Optional training history logs.
            metrics_summary: Optional evaluation metrics to include in metadata.

        Returns:
            Path to the saved directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = self.checkpoints_dir / f"{model_name}_{version}_{timestamp}"
        version_dir.mkdir(parents=True, exist_ok=True)

        app_logger.info(f"Saving checkpoint to {version_dir}...")

        # Save model weights/model in SavedModel format
        model_path = version_dir / "model.keras"
        try:
            model.save(model_path)
            app_logger.info(f"Keras model saved successfully at {model_path}")
        except Exception as e:
            app_logger.error(f"Failed to save Keras model at {model_path}: {e}")

        # Save metadata JSON file
        metadata = {
            "model_name": model_name,
            "version": version,
            "timestamp": timestamp,
            "backbone": settings.MODEL_NAME,
            "image_size": settings.IMAGE_SIZE,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
            "epochs": settings.EPOCHS,
            "batch_size": settings.BATCH_SIZE,
            "learning_rate": settings.LEARNING_RATE,
            "optimizer": settings.OPTIMIZER,
            "loss_function": settings.LOSS_FUNCTION,
            "metrics": metrics_summary or {},
        }

        metadata_path = version_dir / "metadata.json"
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            app_logger.error(f"Failed to write checkpoint metadata to {metadata_path}: {e}")

        # Save history if present
        if history_data:
            history_path = version_dir / "history.json"
            try:
                with open(history_path, "w", encoding="utf-8") as f:
                    json.dump(history_data, f, indent=4)
            except Exception as e:
                app_logger.error(f"Failed to save training history to {history_path}: {e}")

        # Update a 'latest' pointer file
        latest_info = {
            "model_name": model_name,
            "version": version,
            "timestamp": timestamp,
            "path": str(version_dir)
        }
        latest_pointer_path = self.checkpoints_dir / f"latest_{model_name}.json"
        try:
            with open(latest_pointer_path, "w", encoding="utf-8") as f:
                json.dump(latest_info, f, indent=4)
        except Exception as e:
            app_logger.error(f"Failed to update latest pointer for {model_name}: {e}")

        return version_dir

    def load_latest_checkpoint(self, model_name: str) -> Optional[Path]:
        """
        Reads the latest checkpoint path pointer for the given model name.

        Args:
            model_name: Model identifier name.

        Returns:
            Path to the checkpoint folder, or None.
        """
        pointer_path = self.checkpoints_dir / f"latest_{model_name}.json"
        if not pointer_path.exists():
            app_logger.warning(f"No checkpoint pointer found for model {model_name} at {pointer_path}")
            return None

        try:
            with open(pointer_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            checkpoint_path = Path(info.get("path", ""))
            if checkpoint_path.exists():
                app_logger.info(f"Found latest checkpoint for {model_name} at {checkpoint_path}")
                return checkpoint_path
        except Exception as e:
            app_logger.error(f"Error loading checkpoint pointer file: {e}")
        
        return None
