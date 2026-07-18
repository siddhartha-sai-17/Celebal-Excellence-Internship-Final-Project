"""
Module Description: Embeddings Database Unit Tests
Purpose: Assures save, load, and cache updates perform correctly on the database layer.
Author: Technical Lead
Version: 1.0.0
Dependencies: pytest, pandas, numpy, recommendation.embedding_database
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from recommendation.embedding_database import EmbeddingDatabase


@pytest.fixture
def temp_db_dir(tmp_path: Path) -> Path:
    """Returns temporary path folder."""
    return tmp_path / "embeddings"


def test_embedding_database_lifecycle(temp_db_dir: Path) -> None:
    """Tests serialization and deserialization of vectors and metadata."""
    db = EmbeddingDatabase(temp_db_dir)
    assert not db.exists()

    # Create dummy embeddings
    embeddings = np.random.normal(size=(5, 128)).astype(np.float32)
    metadata_df = pd.DataFrame({
        "id": [101, 102, 103, 104, 105],
        "articleType": ["Shirts", "Shirts", "Watches", "Watches", "Jeans"]
    })

    # Save
    db.save(embeddings, metadata_df, {"test_run": True})
    assert db.exists()

    # Load
    loaded_embs, loaded_meta = db.load()
    assert loaded_embs is not None
    assert loaded_meta is not None
    assert loaded_embs.shape == (5, 128)
    assert len(loaded_meta) == 5
    assert np.allclose(loaded_embs, embeddings)
    assert list(loaded_meta["id"]) == [101, 102, 103, 104, 105]

    # Clear cache
    db.clear_cache()
    assert db._embeddings is None
    assert db._metadata_df is None
