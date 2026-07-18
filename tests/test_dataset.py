"""
Module Description: Dataset Preparation Unit Tests
Purpose: Validates positive/negative pair splitting and triplet generation.
Author: Technical Lead
Version: 1.0.0
Dependencies: pytest, pandas, utils.pair_utils, utils.triplet_utils
"""

import pytest
import pandas as pd
from utils.pair_utils import generate_balanced_pairs
from utils.triplet_utils import generate_triplets


@pytest.fixture
def dummy_metadata_df() -> pd.DataFrame:
    """Creates a dummy DataFrame containing different category groups."""
    data = {
        "id": [1, 2, 3, 4, 5, 6],
        "articleType": ["Shirts", "Shirts", "Watches", "Watches", "Jeans", "Jeans"],
        "image_path": [
            "path/1.jpg", "path/2.jpg",
            "path/3.jpg", "path/4.jpg",
            "path/5.jpg", "path/6.jpg"
        ]
    }
    return pd.DataFrame(data)


def test_generate_balanced_pairs(dummy_metadata_df: pd.DataFrame) -> None:
    """Ensures generated pairs are balanced (50% positive, 50% negative)."""
    pairs, labels = generate_balanced_pairs(dummy_metadata_df)
    
    assert len(pairs) == len(labels)
    # Check balance
    pos_count = sum(1 for l in labels if l == 1)
    neg_count = sum(1 for l in labels if l == 0)
    assert pos_count == neg_count
    
    # Assert no duplicate pairs
    unique_pairs = set(pairs)
    assert len(unique_pairs) == len(pairs)


def test_generate_triplets(dummy_metadata_df: pd.DataFrame) -> None:
    """Ensures triplets are built with anchors, positives, and negatives."""
    triplets = generate_triplets(dummy_metadata_df)
    
    assert len(triplets) > 0
    for anchor, positive, negative in triplets:
        # Check category mappings via path indices (path/<id>.jpg)
        anchor_id = int(anchor.split("/")[-1].split(".")[0])
        positive_id = int(positive.split("/")[-1].split(".")[0])
        negative_id = int(negative.split("/")[-1].split(".")[0])
        
        # Resolve categories
        anchor_cat = dummy_metadata_df.loc[dummy_metadata_df["id"] == anchor_id, "articleType"].values[0]
        pos_cat = dummy_metadata_df.loc[dummy_metadata_df["id"] == positive_id, "articleType"].values[0]
        neg_cat = dummy_metadata_df.loc[dummy_metadata_df["id"] == negative_id, "articleType"].values[0]
        
        assert anchor_cat == pos_cat
        assert anchor_cat != neg_cat
        assert anchor != positive  # positive must be distinct from anchor
