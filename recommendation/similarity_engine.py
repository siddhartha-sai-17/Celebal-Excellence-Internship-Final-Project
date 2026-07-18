"""
Module Description: Vectorized Similarity Engine
Purpose: Computes similarity values (Cosine, Dot Product, Euclidean) between query and database embeddings.
Author: Technical Lead
Version: 1.0.0
Dependencies: numpy, sklearn.metrics.pairwise, config.settings, utils.logger
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import settings
from utils.logger import app_logger


class SimilarityEngine:
    """
    Performs vectorized similarity comparisons between high-dimensional features.
    """

    @staticmethod
    def compute_similarity(query_emb: np.ndarray, 
                           database_embs: np.ndarray, 
                           metric: str = settings.SIMILARITY_ALGORITHM) -> np.ndarray:
        """
        Computes pairwise similarity scores between a single query and a database array.

        Args:
            query_emb: 1D or 2D numpy array of shape (embedding_dim,) or (1, embedding_dim).
            database_embs: 2D numpy array of shape (num_samples, embedding_dim).
            metric: The algorithm to run ("cosine", "euclidean", "dot_product").

        Returns:
            A 1D numpy array of similarity scores of shape (num_samples,).
        """
        metric_lower = metric.lower().strip()
        
        # Reshape query to 2D (1, D) if 1D
        if len(query_emb.shape) == 1:
            q_emb = query_emb.reshape(1, -1)
        else:
            q_emb = query_emb

        # Ensure database is 2D
        if len(database_embs.shape) != 2:
            msg = f"Database embeddings must be a 2D matrix. Received shape: {database_embs.shape}"
            app_logger.error(msg)
            raise ValueError(msg)

        if metric_lower == "cosine":
            # Cosine similarity range is [-1, 1]
            scores = cosine_similarity(q_emb, database_embs)[0]
            # Convert cosine similarity to positive distance percentage if necessary
            # (already between -1 and 1, we clip to 0-1 for display convenience)
            scores = np.clip(scores, 0.0, 1.0)
            return scores

        elif metric_lower == "dot_product":
            # Vector dot product
            scores = np.dot(database_embs, q_emb.T).squeeze()
            return scores

        elif metric_lower == "euclidean":
            # Euclidean distance. Similarity = 1 / (1 + distance)
            diff = database_embs - q_emb
            distances = np.linalg.norm(diff, axis=1)
            scores = 1.0 / (1.0 + distances)
            return scores

        else:
            app_logger.warning(f"Unsupported similarity metric '{metric}'. Defaulting to Cosine Similarity.")
            scores = cosine_similarity(q_emb, database_embs)[0]
            return np.clip(scores, 0.0, 1.0)
