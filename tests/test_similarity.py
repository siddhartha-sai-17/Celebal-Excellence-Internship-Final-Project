"""
Module Description: Similarity Search Unit Tests
Purpose: Assures vectorized distance computations (Cosine, Euclidean, Dot Product) return exact matches.
Author: Technical Lead
Version: 1.0.0
Dependencies: pytest, numpy, recommendation.similarity_engine, recommendation.faiss_engine
"""

import pytest
import numpy as np
from recommendation.similarity_engine import SimilarityEngine
from recommendation.faiss_engine import FaissEngine


def test_cosine_similarity() -> None:
    """Validates vectorized cosine metrics on unit vectors."""
    # Identical vectors
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    scores = SimilarityEngine.compute_similarity(v1, v2, metric="cosine")
    assert len(scores) == 2
    assert pytest.approx(scores[0]) == 1.0
    assert pytest.approx(scores[1]) == 0.0


def test_euclidean_distance_similarity() -> None:
    """Validates Euclidean distance mapping to similarity ranges."""
    v1 = np.array([0.0, 0.0])
    v2 = np.array([[0.0, 0.0], [3.0, 4.0]])
    
    # Distance is 0 and 5. Similarities = 1 / (1 + dist) = 1.0 and 1/6
    scores = SimilarityEngine.compute_similarity(v1, v2, metric="euclidean")
    assert pytest.approx(scores[0]) == 1.0
    assert pytest.approx(scores[1]) == (1.0 / 6.0)


def test_faiss_engine_fallback() -> None:
    """Ensures FAISS search yields graceful fallbacks if not built."""
    engine = FaissEngine()
    # Search when index is None should return None
    res = engine.search(np.random.normal(size=(512,)))
    assert res is None
