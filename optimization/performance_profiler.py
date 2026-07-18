"""
Module Description: Performance Profiler Layer
Purpose: Automates latency profile monitoring, saving results to json sheets.
Author: Technical Lead
Version: 1.0.0
Dependencies: json, pathlib, time, config.settings, utils.logger
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from config import settings
from utils.logger import performance_logger


class PerformanceProfiler:
    """
    Profiles latency distributions and records them in JSON sheets.
    """

    def __init__(self, output_dir: Path = settings.REPORTS_DIRECTORY) -> None:
        """
        Initializes the profiler.

        Args:
            output_dir: Folder to save JSON reports.
        """
        self.output_dir: Path = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profile_path: Path = self.output_dir / "performance_profile.json"

    def save_profile(self, metrics: Dict[str, Any]) -> None:
        """
        Exports a performance profile run dictionary as JSON.

        Args:
            metrics: Profile logs.
        """
        try:
            with open(self.profile_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)
            performance_logger.info(f"Performance profile saved successfully to {self.profile_path}")
        except Exception as e:
            performance_logger.error(f"Failed to save performance profile: {e}")
