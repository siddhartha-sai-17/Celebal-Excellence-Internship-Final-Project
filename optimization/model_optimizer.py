"""
Module Description: Model Optimization Wrapper
Purpose: Configures TensorFlow configurations and execution graphs.
Author: Technical Lead
Version: 1.0.0
Dependencies: models.model_utils
"""

from models.model_utils import configure_accelerator, clear_tf_session


class ModelOptimizer:
    """
    Enforces accelerator parameters and dynamic resource allocations.
    """

    @staticmethod
    def initialize_tf_optimizations() -> None:
        """Initializes TF memory allocation options."""
        configure_accelerator()

    @staticmethod
    def optimize_memory_session() -> None:
        """Clears Keras graph definitions."""
        clear_tf_session()
