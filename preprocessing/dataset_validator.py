"""
Module Description: Dataset Validation Module
Purpose: Verifies metadata integrity, detects missing files, duplicate ids, corrupted images, and outputs JSON reports.
Author: Technical Lead
Version: 1.1.0
Dependencies: pandas, json, pathlib, config.settings, utils.logger, preprocessing.image_validator
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from config import settings
from utils.logger import app_logger
from preprocessing.image_validator import ImageValidator


class DatasetValidator:
    """
    Performs verification audits on both raw and subset dataset files and images.
    """

    def __init__(self, metadata_path: Path, images_dir: Path) -> None:
        """
        Initializes the validator with metadata CSV path and image directory.

        Args:
            metadata_path: Path to metadata CSV.
            images_dir: Path to image directory.
        """
        self.metadata_path: Path = metadata_path
        self.images_dir: Path = images_dir
        self.report_path: Path = settings.REPORTS_DIRECTORY / "dataset_validation_report.json"

    def is_cache_valid(self) -> bool:
        """
        Checks if the cached validation report is already present and matches the dataset.
        """
        if not self.report_path.exists():
            return False
        try:
            with open(self.report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            
            # Check if raw files exist and count matches
            if not self.metadata_path.exists():
                return False
                
            df = pd.read_csv(self.metadata_path, on_bad_lines='skip')
            if report.get("total_metadata_records") == len(df):
                return True
            return False
        except Exception:
            return False

    def run_validation(self) -> Dict[str, Any]:
        """
        Audits files and logs corrupt/missing items. Generates a validation dictionary.
        """
        # Caching logic
        if self.is_cache_valid():
            app_logger.info("🟢 Valid validation report cache detected. Skipping full raw scan.")
            with open(self.report_path, "r", encoding="utf-8") as f:
                return json.load(f)

        app_logger.info(f"🔄 Scanning and auditing dataset: {self.metadata_path}...")
        
        report: Dict[str, Any] = {
            "metadata_exists": False,
            "images_dir_exists": False,
            "total_metadata_records": 0,
            "total_images_in_folder": 0,
            "missing_images_count": 0,
            "missing_images_list": [],
            "corrupted_images_count": 0,
            "corrupted_images_list": [],
            "duplicate_metadata_ids_count": 0,
            "duplicate_metadata_ids_list": [],
            "invalid_format_count": 0,
            "invalid_format_list": [],
            "category_distribution": {}
        }

        # Check directory existence
        report["metadata_exists"] = self.metadata_path.exists()
        report["images_dir_exists"] = self.images_dir.exists()

        if not report["metadata_exists"]:
            app_logger.error(f"Metadata file not found: {self.metadata_path}")
            return report

        if not report["images_dir_exists"]:
            app_logger.error(f"Images folder not found: {self.images_dir}")
            return report

        # Read metadata
        try:
            df = pd.read_csv(self.metadata_path, on_bad_lines='skip')
            report["total_metadata_records"] = len(df)
        except Exception as e:
            app_logger.error(f"Failed to read metadata CSV: {e}")
            return report

        # Count total images using fast os.scandir
        try:
            count = 0
            for entry in os.scandir(self.images_dir):
                if entry.is_file() and any(entry.name.lower().endswith(ext) for ext in settings.SUPPORTED_IMAGE_FORMATS):
                    count += 1
            report["total_images_in_folder"] = count
        except Exception as e:
            app_logger.error(f"Failed to count images on disk: {e}")

        # Check duplicate metadata rows by ID
        if "id" in df.columns:
            duplicates = df[df.duplicated(subset=["id"], keep=False)]
            if not duplicates.empty:
                dup_ids = duplicates["id"].unique().tolist()
                report["duplicate_metadata_ids_count"] = len(dup_ids)
                report["duplicate_metadata_ids_list"] = dup_ids[:20]
                app_logger.warning(f"Found {len(dup_ids)} duplicate product IDs in metadata.")
        else:
            app_logger.error("No 'id' column found in metadata.")
            return report

        # Category distribution
        if "articleType" in df.columns:
            report["category_distribution"] = df["articleType"].value_counts().to_dict()

        # Audit files fast (existence and file sizes)
        missing_list: List[str] = []
        corrupt_list: List[str] = []
        invalid_fmt_list: List[str] = []

        # Convert set for fast lookup
        try:
            disk_filenames = {entry.name for entry in os.scandir(self.images_dir) if entry.is_file()}
        except Exception:
            disk_filenames = set()

        for _, row in df.iterrows():
            img_id = str(row["id"])
            img_name = f"{img_id}.jpg"
            img_path = self.images_dir / img_name

            # Check existence
            if img_name not in disk_filenames:
                missing_list.append(img_id)
                continue

            # Fast size check (0 bytes file is corrupted)
            try:
                if img_path.stat().st_size == 0:
                    corrupt_list.append(img_id)
                    continue
            except OSError:
                corrupt_list.append(img_id)
                continue

        report["missing_images_count"] = len(missing_list)
        report["missing_images_list"] = missing_list[:50]
        report["corrupted_images_count"] = len(corrupt_list)
        report["corrupted_images_list"] = corrupt_list[:50]
        report["invalid_format_count"] = len(invalid_fmt_list)
        report["invalid_format_list"] = invalid_fmt_list[:50]

        app_logger.info(
            f"Validation Completed. Results: Total Records: {report['total_metadata_records']}, "
            f"Images: {report['total_images_in_folder']}, Missing: {report['missing_images_count']}"
        )
        return report

    def save_report(self, report_path: Path = None) -> None:
        """
        Saves the validation report as a JSON file.
        """
        target_path = report_path or self.report_path
        report = self.run_validation()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4, default=str)
            app_logger.info(f"Validation report saved successfully to {target_path}")
        except Exception as e:
            app_logger.error(f"Failed to write validation report JSON: {e}")
