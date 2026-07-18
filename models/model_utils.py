"""
Module Description: Model and GPU Utilities
Purpose: Auto-detects accelerators (GPU/CPU) and performs session cleanups to optimize RAM consumption.
Author: Technical Lead
Version: 1.0.0
Dependencies: tensorflow, config.settings, utils.logger
"""

import gc
import tensorflow as tf
from config import settings
from utils.logger import app_logger


def configure_accelerator() -> str:
    """
    Detects GPU and configures memory growth to prevent TensorFlow from allocating all GPU RAM.

    Returns:
        String representing the active device ("GPU" or "CPU").
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Enable dynamic memory allocation
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            app_logger.info(f"GPU device detected: {gpus}. Configured dynamic memory allocation.")
            return "GPU"
        except RuntimeError as e:
            app_logger.error(f"Error configuring GPU memory growth: {e}")
            return "GPU"
    else:
        app_logger.info("No GPU detected. Falling back to CPU execution.")
        return "CPU"


def clear_tf_session() -> None:
    """
    Clears the Keras session and triggers garbage collection to free memory.
    """
    app_logger.info("Clearing TensorFlow session and triggering garbage collection...")
    try:
        tf.keras.backend.clear_session()
        # Trigger explicit python garbage collection
        gc.collect()
        app_logger.info("Session cleared successfully.")
    except Exception as e:
        app_logger.error(f"Failed to clear TensorFlow session: {e}")
