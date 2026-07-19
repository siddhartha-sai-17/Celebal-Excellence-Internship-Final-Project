"""
Module Description: Recommendation Cache Manager
Purpose: Implements in-memory caching of heavy structures (Models, Embeddings, FAISS Indices) to reduce latency.
Author: Technical Lead
Version: 1.0.0
Dependencies: tensorflow, numpy, pandas, pathlib, config.settings, utils.logger
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import tensorflow as tf
from config import settings
from utils.logger import app_logger


class CacheManager:
    """
    Singleton-style container managing global in-memory state of ML assets.
    """
    _instance: Optional["CacheManager"] = None

    # Static dictionaries holding caches
    _models: Dict[str, tf.keras.Model] = {}
    _embeddings: Dict[str, np.ndarray] = {}
    _metadata: Dict[str, pd.DataFrame] = {}
    _faiss_engines: Dict[str, Any] = {}

    def __new__(cls, *args, **kwargs) -> "CacheManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self, source_name: str, checkpoint_path: Optional[Path]) -> Optional[tf.keras.Model]:
        """
        Retrieves a cached model, or loads it if not present.

        Args:
            source_name: Name of embedding source (e.g. 'transfer_learning', 'siamese').
            checkpoint_path: Path to directory of saved weights.

        Returns:
            tf.keras.Model or None.
        """
        cache_key = f"{source_name}_{settings.MODEL_NAME}"
        if cache_key in self._models:
            app_logger.info(f"[Cache Hit] Model {cache_key} retrieved from cache.")
            return self._models[cache_key]

        app_logger.info(f"[Cache Miss] Loading model {cache_key}...")
        from models.model_loader import ModelLoader
        model = ModelLoader.load_embedding_model(
            model_name=settings.MODEL_NAME, 
            checkpoint_dir=checkpoint_path,
            is_baseline=(source_name.lower().strip() == "baseline")
        )
        if model is not None:
            self._models[cache_key] = model
            app_logger.info(f"Model {cache_key} added to memory cache.")
        return model

    def get_database(self, source_name: str, db_dir: Path) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
        """
        Retrieves a cached embedding database (vectors and metadata).

        Args:
            source_name: Source identifier ('baseline', 'transfer_learning', 'siamese').
            db_dir: Folder containing database files.

        Returns:
            A tuple of (embeddings_array, metadata_df).
        """
        if source_name in self._embeddings and source_name in self._metadata:
            app_logger.info(f"[Cache Hit] Embeddings and metadata for '{source_name}' retrieved from cache.")
            return self._embeddings[source_name], self._metadata[source_name]

        app_logger.info(f"[Cache Miss] Loading embedding database for '{source_name}'...")
        from recommendation.embedding_database import EmbeddingDatabase
        db = EmbeddingDatabase(db_dir)
        embs, meta_df = db.load()
        
        if embs is not None and meta_df is not None:
            self._embeddings[source_name] = embs
            self._metadata[source_name] = meta_df
            app_logger.info(f"Embedding database '{source_name}' added to memory cache.")
        return embs, meta_df

    def get_faiss_engine(self, source_name: str, embeddings: np.ndarray) -> Optional[Any]:
        """
        Retrieves a cached FAISS index engine, or builds it.

        Args:
            source_name: Name of active source.
            embeddings: Database embeddings to build index if not found.

        Returns:
            FaissEngine or None.
        """
        if source_name in self._faiss_engines:
            app_logger.info(f"[Cache Hit] FAISS engine for '{source_name}' retrieved from cache.")
            return self._faiss_engines[source_name]

        from recommendation.faiss_engine import FaissEngine
        engine = FaissEngine()
        
        # Try loading first
        loaded = engine.load_index(source_name)
        if not loaded:
            # Build and save
            built = engine.build_index(embeddings, source_name)
            if not built:
                return None

        self._faiss_engines[source_name] = engine
        app_logger.info(f"FAISS engine '{source_name}' added to memory cache.")
        return engine

    def clear_all_caches(self) -> None:
        """Wipes all cache dictionaries to release RAM."""
        self._models.clear()
        self._embeddings.clear()
        self._metadata.clear()
        self._faiss_engines.clear()
        
        # Trigger tf memory session clean
        from models.model_utils import clear_tf_session
        clear_tf_session()
        app_logger.info("All CacheManager memory caches cleared.")
siamese_cache = CacheManager()
