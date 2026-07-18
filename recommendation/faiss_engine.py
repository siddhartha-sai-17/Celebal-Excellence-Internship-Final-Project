"""
Module Description: FAISS Acceleration Engine
Purpose: Builds, loads, saves, and queries FAISS indexes for rapid similarity searches, with automatic cosine fallback.
Author: Technical Lead
Version: 1.0.0
Dependencies: numpy, pathlib, config.settings, utils.logger
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from config import settings
from utils.logger import app_logger

# Optional FAISS import fallback
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    app_logger.warning("FAISS library not installed. Similarity search will fallback to vectorized cosine similarity.")


class FaissEngine:
    """
    Manages FAISS index lifecycle: creation, serialization, querying, and updating.
    """

    def __init__(self, index_dir: Path = settings.FAISS_DIRECTORY) -> None:
        """
        Initializes the FAISS engine.

        Args:
            index_dir: Target directory to save and load FAISS index files.
        """
        self.index_dir: Path = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index: Optional[Any] = None

    def build_index(self, embeddings: np.ndarray, index_name: str) -> bool:
        """
        Builds a FAISS FlatL2 or FlatIP (Inner Product) index for L2 normalized vectors.

        Args:
            embeddings: Numpy array of shape (N, embedding_dim).
            index_name: Name identifier for the index file (e.g. 'baseline').

        Returns:
            True if index build and save succeeded, False otherwise.
        """
        if not FAISS_AVAILABLE:
            return False

        if len(embeddings) == 0:
            app_logger.warning("Cannot build FAISS index with empty embeddings array.")
            return False

        app_logger.info(f"Building FAISS index '{index_name}' for {len(embeddings)} vectors...")
        try:
            dim = embeddings.shape[1]
            
            # Since our embeddings are L2 normalized, Cosine Similarity is equivalent
            # to Inner Product (dot product). IndexFlatIP stands for Inner Product.
            index = faiss.IndexFlatIP(dim)
            
            # Ensure float32 format
            data = embeddings.astype(np.float32)
            index.add(data)
            
            # Save index to disk
            index_path = self.index_dir / f"{index_name}.index"
            faiss.write_index(index, str(index_path))
            self.index = index
            app_logger.info(f"FAISS index built and saved successfully to {index_path}")
            return True
        except Exception as e:
            app_logger.error(f"Failed to build FAISS index: {e}")
            return False

    def load_index(self, index_name: str) -> bool:
        """
        Loads an existing FAISS index from disk.

        Args:
            index_name: Name of index.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not FAISS_AVAILABLE:
            return False

        index_path = self.index_dir / f"{index_name}.index"
        if not index_path.exists():
            app_logger.warning(f"FAISS index file not found at {index_path}")
            return False

        try:
            app_logger.info(f"Loading FAISS index from {index_path}...")
            self.index = faiss.read_index(str(index_path))
            app_logger.info("FAISS index loaded successfully.")
            return True
        except Exception as e:
            app_logger.error(f"Failed to load FAISS index: {e}")
            return False

    def search(self, query_emb: np.ndarray, top_k: int = settings.TOP_K_RESULTS) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Searches the FAISS index for the nearest neighbors.

        Args:
            query_emb: Query vector array of shape (embedding_dim,) or (1, embedding_dim).
            top_k: Number of neighbors to retrieve.

        Returns:
            A tuple of (similarities, indices) or None if search failed or index is not loaded.
        """
        if not FAISS_AVAILABLE or self.index is None:
            return None

        try:
            # Format query shape to 2D
            if len(query_emb.shape) == 1:
                q_emb = query_emb.reshape(1, -1).astype(np.float32)
            else:
                q_emb = query_emb.astype(np.float32)

            # FAISS search
            # D = distances/similarities (Inner Product scores), I = indices of database matches
            similarities, indices = self.index.search(q_emb, top_k)
            return similarities[0], indices[0]
        except Exception as e:
            app_logger.error(f"FAISS index search failed: {e}")
            return None
