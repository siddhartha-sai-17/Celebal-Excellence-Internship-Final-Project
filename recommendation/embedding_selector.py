"""
Module Description: Embedding Selector
Purpose: Configurable selection between baseline, transfer learning, and Siamese embedding databases and weights.
Author: Technical Lead
Version: 1.0.0
Dependencies: pathlib, config.settings, utils.logger
"""

from pathlib import Path
from typing import Tuple, Optional
from config import settings
from utils.logger import app_logger


class EmbeddingSelector:
    """
    Manages resolution of active embedding folder paths and associated model files.
    """

    @staticmethod
    def get_paths(source_name: str = settings.EMBEDDING_SOURCE) -> Tuple[Path, Optional[Path]]:
        """
        Resolves the embedding vector file path and model checkpoint path for the chosen source.

        Args:
            source_name: Name of the embedding source ("baseline", "transfer_learning", "siamese").

        Returns:
            A tuple of (embeddings_dir, model_checkpoint_dir) where:
            - embeddings_dir: Path to directory holding embeddings.npy and metadata.csv.
            - model_checkpoint_dir: Path to model checkpoint directory, or None if baseline (uses ImageNet).
        """
        src_lower = source_name.lower().strip()
        app_logger.info(f"Resolving paths for embedding source: {src_lower}")

        # Target directories
        base_emb_dir = settings.EMBEDDING_DIRECTORY
        checkpoints_dir = settings.MODEL_DIRECTORY / "checkpoints"

        if src_lower == "baseline":
            # Baseline uses raw pre-trained backbone
            emb_dir = base_emb_dir / "baseline"
            emb_dir.mkdir(parents=True, exist_ok=True)
            return emb_dir, None

        elif src_lower == "transfer_learning":
            # Transfer learning embeddings and weights
            emb_dir = base_emb_dir / "transfer_learning"
            emb_dir.mkdir(parents=True, exist_ok=True)
            
            # Look for latest transfer learning checkpoint
            from models.checkpoint_manager import CheckpointManager
            mgr = CheckpointManager(checkpoints_dir)
            latest_path = mgr.load_latest_checkpoint(f"{settings.MODEL_NAME}_transfer")
            return emb_dir, latest_path

        elif src_lower == "siamese":
            # Siamese embeddings and weights
            emb_dir = base_emb_dir / "siamese"
            emb_dir.mkdir(parents=True, exist_ok=True)
            
            # Look for latest siamese checkpoint
            from models.checkpoint_manager import CheckpointManager
            mgr = CheckpointManager(checkpoints_dir)
            latest_path = mgr.load_latest_checkpoint(f"{settings.MODEL_NAME}_siamese")
            return emb_dir, latest_path

        else:
            msg = f"Unknown embedding source name: {source_name}. Defaulting to 'baseline'."
            app_logger.warning(msg)
            # Default fallback
            emb_dir = base_emb_dir / "baseline"
            emb_dir.mkdir(parents=True, exist_ok=True)
            return emb_dir, None
