"""
Module Description: System Startup Validator
Purpose: Audits configuration files, folders, metadata, and models during startup, generating validation logs.
Author: Technical Lead
Version: 1.0.0
Dependencies: json, pathlib, config.settings, utils.logger
"""

import json
from pathlib import Path
from typing import Dict, Any
from config import settings
from utils.logger import app_logger


class SystemValidator:
    """
    System integrity check covering folders, dataset metadata, embeddings, and model weights.
    """

    def __init__(self, output_dir: Path = settings.REPORTS_DIRECTORY) -> None:
        """
        Initializes the validator.

        Args:
            output_dir: Target folder to save logs.
        """
        self.output_dir: Path = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_path: Path = self.output_dir / "system_validation.json"

    def run_validation_checks(self) -> Dict[str, Any]:
        """
        Performs audits on configuration parameters and directory states.

        Returns:
            Dictionary showing status of folders, datasets, and models.
        """
        app_logger.info("Running system startup validation checks...")
        
        status: Dict[str, Any] = {
            "dirs": {},
            "files": {},
            "models": {},
            "healthy": True
        }

        # 1. Verify directory layout
        dirs_to_check = {
            "dataset": settings.DATASET_DIRECTORY,
            "subset": settings.SUBSET_DIRECTORY,
            "models": settings.MODEL_DIRECTORY,
            "embeddings": settings.EMBEDDING_DIRECTORY,
            "logs": settings.LOG_DIRECTORY,
            "evaluation": settings.EVALUATION_DIRECTORY
        }

        for label, path in dirs_to_check.items():
            exists = path.exists()
            status["dirs"][label] = exists
            if not exists:
                status["healthy"] = False

        # 2. Check metadata CSV files
        styles_csv = settings.DATASET_DIRECTORY / "styles.csv"
        subset_metadata = settings.SUBSET_DIRECTORY / "subset_metadata.csv"
        status["files"]["raw_styles_csv"] = styles_csv.exists()
        status["files"]["subset_metadata_csv"] = subset_metadata.exists()

        # 3. Check embedding databases on disk
        base_dir = settings.EMBEDDING_DIRECTORY
        status["files"]["baseline_embeddings"] = (base_dir / "baseline" / "embeddings.npy").exists()
        status["files"]["transfer_embeddings"] = (base_dir / "transfer_learning" / "embeddings.npy").exists()
        status["files"]["siamese_embeddings"] = (base_dir / "siamese" / "embeddings.npy").exists()

        # 4. Check model checkpoints
        checkpoints_dir = settings.MODEL_DIRECTORY / "checkpoints"
        from models.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager(checkpoints_dir)
        status["models"]["transfer_model_latest"] = mgr.load_latest_checkpoint(f"{settings.MODEL_NAME}_transfer") is not None
        status["models"]["siamese_model_latest"] = mgr.load_latest_checkpoint(f"{settings.MODEL_NAME}_siamese") is not None

        # Save verification report
        try:
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, default=str)
            app_logger.info(f"System validation log written to {self.report_path}")
        except Exception as e:
            app_logger.error(f"Failed to save system validation log: {e}")

        return status
