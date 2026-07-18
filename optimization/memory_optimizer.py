"""
Module Description: Memory Optimizer Layer
Purpose: Monitors and optimizes memory footprint (RAM and VRAM) by forcing garbage collection and clearing sessions.
Author: Technical Lead
Version: 1.0.0
Dependencies: os, gc, tensorflow, utils.logger
"""

import os
import gc
import tensorflow as tf
from utils.logger import performance_logger


class MemoryOptimizer:
    """
    Tracks RAM consumption and safely clears GPU/CPU memory structures when threshold limits are breached.
    """

    @staticmethod
    def get_process_memory_mb() -> float:
        """
        Queries the current OS process memory consumption in Megabytes.
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            # Fallback estimation
            return 0.0

    @classmethod
    def force_garbage_collection(cls) -> float:
        """
        Forces a Python garbage collection cycle and clears Keras sessions.

        Returns:
            Float representing the amount of memory freed in MB.
        """
        mem_before = cls.get_process_memory_mb()
        performance_logger.info(f"RAM consumption before cleanup: {mem_before:.2f}MB")

        # 1. Clear Keras/TF session backend
        try:
            tf.keras.backend.clear_session()
        except Exception as e:
            performance_logger.warning(f"Keras session clear failed: {e}")

        # 2. Collect garbage
        collected = gc.collect()
        
        mem_after = cls.get_process_memory_mb()
        freed = max(0.0, mem_before - mem_after)
        
        performance_logger.info(
            f"Garbage collection forced. Collected objects: {collected}. "
            f"RAM consumption after cleanup: {mem_after:.2f}MB (Freed: {freed:.2f}MB)"
        )
        return freed

    @classmethod
    def log_system_memory_report(cls) -> None:
        """Logs the current process memory metrics to performance logs."""
        mem = cls.get_process_memory_mb()
        performance_logger.info(f"[Memory Report] Current RAM Usage: {mem:.2f}MB")
