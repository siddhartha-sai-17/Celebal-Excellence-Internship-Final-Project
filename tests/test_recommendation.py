"""
Module Description: Recommendation Engine Unit Tests
Purpose: Assures ranking filters, deduplication, and metadata parsing perform correctly.
Author: Technical Lead
Version: 1.0.0
Dependencies: pytest, pandas, numpy, recommendation.ranking_engine, recommendation.metadata_loader
"""

import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from recommendation.ranking_engine import RankingEngine
from recommendation.metadata_loader import MetadataLoader


@pytest.fixture
def dummy_metadata_df() -> pd.DataFrame:
    """Creates a dummy DataFrame containing details."""
    data = {
        "id": [101, 102, 103, 104, 105],
        "gender": ["Men", "Women", "Men", "Women", "Men"],
        "articleType": ["Shirts", "Shirts", "Watches", "Watches", "Jeans"],
        "subCategory": ["Topwear", "Topwear", "Watches", "Watches", "Bottomwear"],
        "baseColour": ["Navy Blue", "Blue", "Black", "Black", "Black"],
        "season": ["Fall", "Summer", "Winter", "Winter", "Fall"],
        "usage": ["Casual", "Casual", "Casual", "Casual", "Casual"],
        "displayName": ["Shirt A", "Shirt B", "Watch A", "Watch B", "Jeans A"],
        "image_path": ["path/1.jpg", "path/2.jpg", "path/3.jpg", "path/4.jpg", "path/5.jpg"]
    }
    return pd.DataFrame(data)


def test_ranking_engine_sort(dummy_metadata_df: pd.DataFrame) -> None:
    """Ensures that scores are ranked in descending order."""
    # Dummy similarity scores
    scores = np.array([0.45, 0.90, 0.72, 0.15, 0.81])
    
    results = RankingEngine.rank_and_filter(
        scores=scores,
        metadata_df=dummy_metadata_df,
        similarity_threshold=0.2,  # Excludes index 3 (score 0.15)
        top_k=5
    )

    # 4 results match threshold
    assert len(results) == 4
    
    # Check descending order of scores
    assert results[0]["similarity_score"] == 0.90  # Shirt B (index 1)
    assert results[1]["similarity_score"] == 0.81  # Jeans A (index 4)
    assert results[2]["similarity_score"] == 0.72  # Watch A (index 2)
    assert results[3]["similarity_score"] == 0.45  # Shirt A (index 0)


def test_ranking_engine_filters(dummy_metadata_df: pd.DataFrame) -> None:
    """Ensures gender and category filters restrict matches correctly."""
    scores = np.array([0.80, 0.90, 0.70, 0.60, 0.50])
    
    # 1. Gender filter: Men
    results = RankingEngine.rank_and_filter(
        scores=scores,
        metadata_df=dummy_metadata_df,
        gender_filter="Men",
        top_k=5
    )
    # Men IDs are 101, 103, 105
    for item in results:
        assert item["gender"] == "Men"

    # 2. Category filter: Watches
    results = RankingEngine.rank_and_filter(
        scores=scores,
        metadata_df=dummy_metadata_df,
        category_filter="Watches",
        top_k=5
    )
    assert len(results) == 2
    assert results[0]["id"] == "104" or results[0]["id"] == "103"


def test_metadata_loader(tmp_path: Path, dummy_metadata_df: pd.DataFrame) -> None:
    """Ensures metadata loader loads uniques for filters dropdowns."""
    csv_path = tmp_path / "styles.csv"
    dummy_metadata_df.to_csv(csv_path, index=False)

    loader = MetadataLoader(csv_path)
    options = loader.get_filter_options()
    
    assert "genders" in options
    assert "categories" in options
    assert options["genders"] == ["Men", "Women"]
    assert options["categories"] == ["Jeans", "Shirts", "Watches"]  # sorted alphabetically
