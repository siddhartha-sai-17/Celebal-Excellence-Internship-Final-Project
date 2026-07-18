"""
Module Description: Embeddings Database Manager
Purpose: Saves, loads, and manages NumPy arrays of image embeddings alongside metadata CSV mappings.
Author: Technical Lead
Version: 1.0.0
Dependencies: numpy, pandas, json, pathlib, datetime, config.settings, utils.logger
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from config import settings
from utils.logger import app_logger


class EmbeddingDatabase:
    """
    Manages embedding serialization, retrieval, metadata alignment, and update checks.
    """

    def __init__(self, db_dir: Path) -> None:
        """
        Initializes the database directory.

        Args:
            db_dir: Folder containing the specific embedding files.
        """
        self.db_dir: Path = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings_path: Path = self.db_dir / "embeddings.npy"
        self.metadata_path: Path = self.db_dir / "metadata.csv"
        self.config_path: Path = self.db_dir / "embedding_config.json"

        # Cached in-memory objects
        self._embeddings: Optional[np.ndarray] = None
        self._metadata_df: Optional[pd.DataFrame] = None

    def exists(self) -> bool:
        """Checks if both the embeddings array and metadata files exist."""
        return self.embeddings_path.exists() and self.metadata_path.exists()

    def save(self, embeddings: np.ndarray, metadata_df: pd.DataFrame, extra_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Saves the embeddings array, metadata DataFrame, and generates a config log.

        Args:
            embeddings: Numpy array of shape (N, embedding_dim).
            metadata_df: DataFrame containing image IDs and details (len N).
            extra_config: Optional metadata dictionary to save in config.
        """
        app_logger.info(f"Saving embeddings database to {self.db_dir}...")
        
        try:
            # Save numpy array
            np.save(str(self.embeddings_path), embeddings.astype(np.float32))
            
            # Save metadata dataframe
            metadata_df.to_csv(self.metadata_path, index=False)
            
            # Save info configuration
            info = {
                "timestamp": datetime.now().isoformat(),
                "num_embeddings": len(embeddings),
                "embedding_dimension": embeddings.shape[1] if len(embeddings) > 0 else 0,
                "model_name": settings.MODEL_NAME,
                "dataset_size": len(metadata_df)
            }
            if extra_config:
                info.update(extra_config)

            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=4)

            # Update cache
            self._embeddings = embeddings
            self._metadata_df = metadata_df
            app_logger.info(f"Saved {len(embeddings)} embeddings successfully.")
        except Exception as e:
            app_logger.error(f"Failed to save embeddings database: {e}")

    def load(self) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
        """
        Loads the embeddings array and metadata DataFrame from disk or memory cache.

        Returns:
            A tuple of (embeddings_array, metadata_df).
        """
        if self._embeddings is not None and self._metadata_df is not None:
            return self._embeddings, self._metadata_df

        if not self.exists():
            app_logger.warning(f"Embedding database files do not exist at {self.db_dir}")
            return None, None

        app_logger.info(f"Loading embeddings database from {self.db_dir}...")
        try:
            # Load numpy array
            self._embeddings = np.load(str(self.embeddings_path))
            
            # Load metadata dataframe
            self._metadata_df = pd.read_csv(self.metadata_path, on_bad_lines='skip')
            
            app_logger.info(f"Loaded {len(self._embeddings)} embeddings from disk successfully.")
            return self._embeddings, self._metadata_df
        except Exception as e:
            app_logger.error(f"Failed to load embeddings database: {e}")
            return None, None

    def clear_cache(self) -> None:
        """Clears in-memory references to free up memory resources."""
        self._embeddings = None
        self._metadata_df = None
        app_logger.info(f"Cleared in-memory cache for embedding database: {self.db_dir}")
