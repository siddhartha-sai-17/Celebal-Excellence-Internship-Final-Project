"""
Module Description: Dataset Subsetting and Unpacking
Purpose: Extracts a balanced category subset from raw project directories or a zip fallback, caching results for instant startup.
Author: Technical Lead
Version: 1.1.0
Dependencies: pandas, zipfile, pathlib, shutil, config.settings, utils.logger
"""

import os
import random
import shutil
import zipfile
import json
from pathlib import Path
from typing import List, Set, Dict, Any, Tuple
import pandas as pd
from config import settings
from utils.logger import app_logger
from utils.timer import time_function


class DatasetSubsetGenerator:
    """
    Handles checking raw dataset folders, sampling a balanced subset, and copying
    subset images to data/subset/images/ while caching configurations for performance.
    """

    def __init__(self, zip_path: Path = Path(r"C:\Users\banda\Downloads\archive (3).zip")) -> None:
        """
        Initializes path mappings dynamically using settings BASE_DIR.

        Args:
            zip_path: Fallback path to the raw dataset zip archive.
        """
        self.zip_path: Path = zip_path
        self.dataset_dir: Path = settings.DATASET_DIRECTORY
        self.raw_images_dir: Path = self.dataset_dir / "images"
        self.raw_styles_csv: Path = self.dataset_dir / "styles.csv"

        self.subset_dir: Path = settings.SUBSET_DIRECTORY
        self.subset_images_dir: Path = self.subset_dir / "images"

        # Create target directories
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.subset_dir.mkdir(parents=True, exist_ok=True)
        self.subset_images_dir.mkdir(parents=True, exist_ok=True)

    def is_cache_valid(self) -> bool:
        """
        Verifies if the already generated subset cache matches the active configurations.
        """
        subset_csv = self.subset_dir / "subset_metadata.csv"
        stats_file = settings.EVALUATION_DIRECTORY / "dataset_statistics.json"

        if not subset_csv.exists() or not stats_file.exists():
            return False

        try:
            # Check cached stats configuration
            with open(stats_file, "r", encoding="utf-8") as f:
                stats = json.load(f)
            
            cached_categories = stats.get("categories", [])
            cached_count = stats.get("total_images", 0)

            # Match active settings
            if set(cached_categories) != set(settings.SELECTED_CATEGORIES):
                return False

            # Verify images exist on disk
            df = pd.read_csv(subset_csv)
            for idx, row in df.iterrows():
                if not (settings.BASE_DIR / row["image_path"]).exists():
                    return False

            return True
        except Exception:
            return False

    @time_function("subset_generation")
    def generate_subset(self) -> pd.DataFrame:
        """
        Samples a balanced subset from the local raw dataset or zip file and registers metadata.
        """
        # Fast caching check
        if self.is_cache_valid():
            app_logger.info("🟢 Valid subset cache detected. Skipping subset image copying.")
            subset_csv = self.subset_dir / "subset_metadata.csv"
            return pd.read_csv(subset_csv)

        app_logger.info("🔄 Cache invalid or missing. Generating balanced subset...")

        # 1. Resolve raw metadata styles.csv
        if not self.raw_styles_csv.exists():
            if self.zip_path.exists():
                app_logger.info(f"Extracting styles.csv from fallback zip archive {self.zip_path}...")
                with zipfile.ZipFile(self.zip_path, 'r') as z:
                    csv_name = next((n for n in z.namelist() if n.endswith("styles.csv")), None)
                    if csv_name:
                        temp_csv = Path(z.extract(csv_name, self.dataset_dir))
                        shutil.move(str(temp_csv), str(self.raw_styles_csv))
                    else:
                        raise FileNotFoundError("styles.csv not found in zip fallback archive.")
            else:
                raise FileNotFoundError(f"Raw metadata file not found at {self.raw_styles_csv} and zip fallback is missing.")

        # 2. Load and clean metadata
        df = pd.read_csv(self.raw_styles_csv, on_bad_lines='skip')
        if 'productDisplayName' in df.columns:
            df = df.rename(columns={'productDisplayName': 'displayName'})

        # Filter categories
        categories = settings.SELECTED_CATEGORIES
        df_filtered = df[df["articleType"].isin(categories)].copy()
        
        # Sample balanced rows
        sampled_dfs = []
        for cat in categories:
            cat_df = df_filtered[df_filtered["articleType"] == cat]
            sample_size = min(settings.SAMPLES_PER_CATEGORY, len(cat_df))
            sampled_df = cat_df.sample(n=sample_size, random_state=settings.RANDOM_SEED)
            sampled_dfs.append(sampled_df)

        df_subset = pd.concat(sampled_dfs, ignore_index=True)
        selected_ids = set(df_subset["id"].astype(str).tolist())

        # 3. Extract or copy images
        copied_count = 0

        # Try copying from raw folder first
        if self.raw_images_dir.exists() and any(self.raw_images_dir.iterdir()):
            app_logger.info(f"Copying {len(selected_ids)} images from raw directory: {self.raw_images_dir}")
            for img_id in selected_ids:
                src_path = self.raw_images_dir / f"{img_id}.jpg"
                if src_path.exists():
                    dest_path = self.subset_images_dir / f"{img_id}.jpg"
                    shutil.copy2(src_path, dest_path)
                    copied_count += 1
        
        # Try extracting from zip archive
        elif self.zip_path.exists():
            app_logger.info(f"Extracting {len(selected_ids)} images from zip archive fallback...")
            with zipfile.ZipFile(self.zip_path, 'r') as z:
                for name in z.namelist():
                    if "images/" in name and any(name.endswith(ext) for ext in settings.SUPPORTED_IMAGE_FORMATS):
                        img_id = Path(name).stem
                        if img_id in selected_ids:
                            with z.open(name) as src_file:
                                dest_path = self.subset_images_dir / f"{img_id}.jpg"
                                with open(dest_path, "wb") as f_out:
                                    shutil.copyfileobj(src_file, f_out)
                                copied_count += 1
        else:
            raise FileNotFoundError(
                f"No source images found in {self.raw_images_dir} and fallback zip archive is missing."
            )

        app_logger.info(f"Successfully processed {copied_count} subset images.")

        # Update subset file path pointers (relative paths resolved using base workspace root)
        df_subset["image_path"] = df_subset["id"].apply(
            lambda idx: str((self.subset_images_dir / f"{idx}.jpg").relative_to(settings.BASE_DIR))
        )

        # Save subset metadata files
        subset_csv_path = self.subset_dir / "subset_metadata.csv"
        df_subset.to_csv(subset_csv_path, index=False)
        df_subset.to_csv(self.dataset_dir / "styles_subset.csv", index=False)

        # Save stats JSON
        stats = {
            "total_images": len(df_subset),
            "categories": settings.SELECTED_CATEGORIES,
            "images_per_category": df_subset["articleType"].value_counts().to_dict()
        }
        stats_path = settings.EVALUATION_DIRECTORY / "dataset_statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)

        return df_subset
