"""
Module Description: Metadata Loader Utility
Purpose: Safely reads metadata CSVs and exposes unique category lists for frontend navigation and filtering.
Author: Technical Lead
Version: 1.0.0
Dependencies: pandas, config.settings, utils.logger
"""

from pathlib import Path
from typing import Dict, List, Set, Optional
import pandas as pd
from config import settings
from utils.logger import app_logger


class MetadataLoader:
    """
    Helper class to load, clean, and parse product metadata columns for search dropdown filters.
    """

    def __init__(self, csv_path: Path) -> None:
        """
        Initializes the loader.

        Args:
            csv_path: Path to the metadata CSV file.
        """
        self.csv_path: Path = csv_path
        self.df: Optional[pd.DataFrame] = None

    def load_metadata(self) -> Optional[pd.DataFrame]:
        """
        Loads and returns the metadata DataFrame, cleaning column names and bad lines.

        Returns:
            DataFrame of metadata or None.
        """
        if self.df is not None:
            return self.df

        if not self.csv_path.exists():
            app_logger.error(f"Metadata file not found: {self.csv_path}")
            return None

        try:
            # Safe reading with error row omission
            self.df = pd.read_csv(self.csv_path, on_bad_lines='skip')
            
            # Map productDisplayName to displayName
            if 'productDisplayName' in self.df.columns:
                self.df = self.df.rename(columns={'productDisplayName': 'displayName'})
                
            # Clean missing values
            self.df = self.df.fillna("Unknown")
            app_logger.info(f"Loaded {len(self.df)} metadata rows successfully from {self.csv_path}")
            return self.df
        except Exception as e:
            app_logger.error(f"Failed to load metadata file: {e}")
            return None

    def get_filter_options(self) -> Dict[str, List[str]]:
        """
        Extracts lists of unique options for categorical metadata columns.

        Returns:
            Dictionary containing sorted lists of unique attribute values.
        """
        df = self.load_metadata()
        options: Dict[str, List[str]] = {
            "genders": [],
            "categories": [],
            "colors": [],
            "seasons": [],
            "usages": []
        }

        if df is None:
            return options

        try:
            # Helper to extract and sort unique values
            def get_sorted_uniques(col: str) -> List[str]:
                if col in df.columns:
                    return sorted([str(x) for x in df[col].unique() if x != "Unknown"])
                return []

            options["genders"] = get_sorted_uniques("gender")
            options["categories"] = get_sorted_uniques("articleType")
            options["colors"] = get_sorted_uniques("baseColour")
            options["seasons"] = get_sorted_uniques("season")
            options["usages"] = get_sorted_uniques("usage")

        except Exception as e:
            app_logger.error(f"Error parsing metadata filter options: {e}")

        return options
