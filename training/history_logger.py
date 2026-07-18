"""
Module Description: History Logging Utility
Purpose: Serializes training history curves to JSON and CSV formats for subsequent rendering.
Author: Technical Lead
Version: 1.0.0
Dependencies: json, pandas, pathlib, utils.logger
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from utils.logger import training_logger


class HistoryLogger:
    """
    Serializes and saves Keras training curves to files.
    """

    @staticmethod
    def save_history(history_dict: Dict[str, List[float]], output_dir: Path) -> None:
        """
        Saves accuracy and loss logs to history.json and history.csv inside output_dir.

        Args:
            history_dict: Dictionary containing historical logs.
            output_dir: Folder path where files will be stored.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        training_logger.info(f"Saving training history curves to {output_dir}...")

        # 1. Save JSON
        json_path = output_dir / "history.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(history_dict, f, indent=4)
            training_logger.info(f"Saved history JSON to {json_path}")
        except Exception as e:
            training_logger.error(f"Failed to save history JSON: {e}")

        # 2. Save CSV
        csv_path = output_dir / "history.csv"
        try:
            df = pd.DataFrame(history_dict)
            df.to_csv(csv_path, index_label="epoch")
            training_logger.info(f"Saved history CSV to {csv_path}")
        except Exception as e:
            training_logger.error(f"Failed to save history CSV: {e}")
