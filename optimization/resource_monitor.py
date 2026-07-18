"""
Module Description: Resource Monitoring Layer
Purpose: Gathers CPU, memory, disk, active threads, and GPU metrics for real-time dashboard telemetry.
Author: Technical Lead
Version: 1.0.0
Dependencies: psutil, threading, os, utils.logger
"""

import os
import sys
import threading
from typing import Dict, Any
from utils.logger import performance_logger

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class ResourceMonitor:
    """
    Captures runtime telemetry metrics for performance charts.
    """

    @staticmethod
    def get_system_metrics() -> Dict[str, Any]:
        """
        Compiles resource usage dictionary.

        Returns:
            Dictionary containing CPU, RAM, Disk, and Thread statistics.
        """
        metrics: Dict[str, Any] = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "memory_used_gb": 0.0,
            "memory_total_gb": 0.0,
            "disk_percent": 0.0,
            "disk_free_gb": 0.0,
            "active_threads": threading.active_count(),
            "gpu_available": False,
            "gpu_name": "N/A"
        }

        # Check if psutil is available
        if PSUTIL_AVAILABLE:
            try:
                metrics["cpu_percent"] = float(psutil.cpu_percent(interval=None))
                
                mem = psutil.virtual_memory()
                metrics["memory_percent"] = float(mem.percent)
                metrics["memory_used_gb"] = float(round(mem.used / (1024 ** 3), 2))
                metrics["memory_total_gb"] = float(round(mem.total / (1024 ** 3), 2))
                
                disk = psutil.disk_usage(os.getcwd())
                metrics["disk_percent"] = float(disk.percent)
                metrics["disk_free_gb"] = float(round(disk.free / (1024 ** 3), 2))
            except Exception as e:
                performance_logger.warning(f"Error gathering psutil metrics: {e}")

        # Check GPU availability using TensorFlow
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                metrics["gpu_available"] = True
                metrics["gpu_name"] = "TensorFlow GPU Accelerator"
        except Exception:
            pass

        return metrics
