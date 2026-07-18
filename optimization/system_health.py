"""
Module Description: System Health Check
Purpose: Evaluates telemetry levels for RAM, CPU, disk, and log health.
Author: Technical Lead
Version: 1.0.0
Dependencies: optimization.resource_monitor, config.settings, utils.logger
"""

from typing import Dict, Any
from config import settings
from utils.logger import app_logger
from optimization.resource_monitor import ResourceMonitor


class SystemHealth:
    """
    Performs quick heartbeats of system status indicators.
    """

    @staticmethod
    def get_health_status() -> Dict[str, Any]:
        """
        Queries resource monitor and flags warnings if CPU/RAM consumption is high.

        Returns:
            Dictionary indicating system health metrics and flags.
        """
        metrics = ResourceMonitor.get_system_metrics()
        
        health_status = {
            "status": "Healthy",
            "cpu_warning": False,
            "memory_warning": False,
            "disk_warning": False,
            "metrics": metrics
        }

        # Check warnings thresholds (e.g. 90%)
        if metrics["cpu_percent"] > 90.0:
            health_status["cpu_warning"] = True
            health_status["status"] = "Warning"
            app_logger.warning(f"High CPU utilization: {metrics['cpu_percent']}%")

        if metrics["memory_percent"] > 90.0:
            health_status["memory_warning"] = True
            health_status["status"] = "Warning"
            app_logger.warning(f"High Memory consumption: {metrics['memory_percent']}%")

        if metrics["disk_percent"] > 90.0:
            health_status["disk_warning"] = True
            health_status["status"] = "Warning"
            app_logger.warning(f"Low Disk Space: {metrics['disk_percent']}% used")

        return health_status
