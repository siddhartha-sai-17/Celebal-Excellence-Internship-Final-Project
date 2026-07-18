"""
Module Description: Training Callbacks Manager
Purpose: Instantiates Keras training callbacks: checkpointing, early stopping, and schedulers.
Author: Technical Lead
Version: 1.0.0
Dependencies: tensorflow, config.settings, utils.logger
"""

import os
from pathlib import Path
from typing import List
import tensorflow as tf
from config import settings
from utils.logger import training_logger


def get_callbacks(checkpoint_dir: Path, 
                  monitor_metric: str = "val_loss", 
                  patience: int = settings.EARLY_STOPPING_PATIENCE) -> List[tf.keras.callbacks.Callback]:
    """
    Instantiates standard callbacks for the training process.

    Args:
        checkpoint_dir: Path to directory where checkpoints are stored.
        monitor_metric: Validation metric to monitor.
        patience: Epochs to wait for validation improvements before terminating.

    Returns:
        List of tf.keras.callbacks.Callback elements.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_filepath = checkpoint_dir / "best_weights.weights.h5"
    csv_log_filepath = checkpoint_dir / "training_log.csv"

    training_logger.info(f"Setting up training callbacks at {checkpoint_dir}...")

    # 1. Model Checkpoint (save weights only for functional compatibility)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(weights_filepath),
        monitor=monitor_metric,
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    # 2. Early Stopping
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor=monitor_metric,
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    # 3. Learning Rate Scheduler (Reduce on Plateau)
    lr_scheduler_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=0.2,
        patience=max(2, patience // 2),
        min_lr=1e-7,
        verbose=1
    )

    # 4. CSV Logger
    csv_logger_cb = tf.keras.callbacks.CSVLogger(
        filename=str(csv_log_filepath),
        separator=",",
        append=False
    )

    # 5. Terminate on NaN values
    terminate_nan_cb = tf.keras.callbacks.TerminateOnNaN()

    return [checkpoint_cb, early_stopping_cb, lr_scheduler_cb, csv_logger_cb, terminate_nan_cb]
