"""
Module Description: Siamese Network Training Coordinator
Purpose: Orchestrates the training of the Siamese metric learning model using pre-trained Transfer Learning weights.
Author: Technical Lead
Version: 1.0.0
Dependencies: pandas, numpy, tensorflow, sklearn, config.settings, utils.logger, siamese.model, siamese.losses, siamese.dataset, utils.pair_utils, utils.triplet_utils
"""

import time
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from config import settings
from utils.logger import training_logger, exception_logger
from utils.timer import time_function
from models.model_factory import ModelFactory
from models.checkpoint_manager import CheckpointManager
from models.model_loader import ModelLoader
from models.model_utils import configure_accelerator
from siamese.model import SiameseModel
from siamese.losses import ContrastiveLoss, TripletLoss
from siamese.dataset import SiameseDatasetBuilder
from utils.pair_utils import generate_balanced_pairs
from utils.triplet_utils import generate_triplets
from training.callbacks import get_callbacks
from training.history_logger import HistoryLogger
from utils.visualization import plot_training_curves


class SiameseTrainer:
    """
    Coordinates training pairs/triplets selection, metric learning loss compilation, and model export.
    """

    def __init__(self, subset_metadata_path: Path = settings.SUBSET_DIRECTORY / "subset_metadata.csv") -> None:
        """
        Initializes the trainer.

        Args:
            subset_metadata_path: Path to subset_metadata.csv.
        """
        self.subset_metadata_path: Path = subset_metadata_path
        self.checkpoints_dir: Path = settings.MODEL_DIRECTORY / "checkpoints"
        self.checkpoint_manager = CheckpointManager(self.checkpoints_dir)
        
        # Configure GPU/CPU dynamic allocation
        self.device = configure_accelerator()

    @time_function("siamese_network_training")
    def run_training(self, version: str = "v1") -> Tuple[tf.keras.Model, Dict[str, Any]]:
        """
        Runs the Siamese training pipeline: loads base model, builds Siamese wrapper, trains, and saves.

        Args:
            version: Target version identifier.

        Returns:
            A tuple of (trained_embedding_model, history_metrics).
        """
        training_logger.info("Initializing Siamese training pipeline...")

        # 1. Load subset metadata
        if not self.subset_metadata_path.exists():
            msg = f"Subset metadata file not found at {self.subset_metadata_path}. Please generate subset first."
            training_logger.error(msg)
            raise FileNotFoundError(msg)

        df = pd.read_csv(self.subset_metadata_path, on_bad_lines='skip')
        
        # Filter rows with existing image files
        valid_rows = []
        for idx, row in df.iterrows():
            img_path = Path(row["image_path"])
            if not img_path.is_absolute():
                img_path = settings.BASE_DIR / img_path
            if img_path.exists():
                row["image_path"] = str(img_path)
                valid_rows.append(row)
        df_valid = pd.DataFrame(valid_rows)
        training_logger.info(f"Loaded {len(df_valid)} valid subset image rows for Siamese training.")

        # 2. Build or Load the Pre-trained Embedding model
        # Try loading Transfer Learning weights first
        mgr_tl = CheckpointManager(self.checkpoints_dir)
        latest_tl_checkpoint = mgr_tl.load_latest_checkpoint(f"{settings.MODEL_NAME}_transfer")

        if latest_tl_checkpoint is not None:
            training_logger.info(f"Initializing Siamese backbone with fine-tuned Transfer Learning weights from {latest_tl_checkpoint}")
            base_model = ModelLoader.load_embedding_model(
                model_name=settings.MODEL_NAME,
                checkpoint_dir=latest_tl_checkpoint
            )
        else:
            training_logger.warning("No fine-tuned Transfer Learning weights found. Initializing with ImageNet weights...")
            base_model = ModelFactory.build_embedding_model(
                model_name=settings.MODEL_NAME,
                embedding_dim=settings.EMBEDDING_DIMENSION,
                dropout_rate=settings.DROPOUT_RATE,
                freeze_backbone=False  # Make it false so we can fine tune
            )

        if base_model is None:
            msg = "Failed to build base embedding model for Siamese training."
            training_logger.error(msg)
            raise ValueError(msg)

        # Freeze the backbone layer inside the base model for CPU performance optimization
        for layer in base_model.layers:
            if layer.name in [settings.MODEL_NAME, "resnet50", "efficientnetb0"] or "resnet" in layer.name.lower() or "efficient" in layer.name.lower():
                layer.trainable = False
                training_logger.info(f"Froze backbone layer '{layer.name}' for fast Siamese training.")

        # 3. Preload all unique images to RAM
        all_unique_paths = df_valid["image_path"].unique().tolist()
        SiameseDatasetBuilder.preload_all_images(all_unique_paths)

        # 4. Handle Loss Function Selection (Contrastive or Triplet)
        loss_name = settings.SIAMESE_LOSS_FUNCTION.lower().strip()
        
        if loss_name == "contrastive":
            training_logger.info("Setting up Contrastive Loss Siamese Pair training...")
            
            # Generate pairs
            pairs, labels = generate_balanced_pairs(df_valid)
            
            # Split train/val (80/20)
            train_pairs, val_pairs, train_labels, val_labels = train_test_split(
                pairs, labels, test_size=0.2, random_state=settings.RANDOM_SEED, stratify=labels
            )

            # Build tf.data datasets
            train_ds = SiameseDatasetBuilder.build_pair_dataset(train_pairs, train_labels, is_training=True)
            val_ds = SiameseDatasetBuilder.build_pair_dataset(val_pairs, val_labels, is_training=False)

            # Build Siamese model wrapper
            siamese_model = SiameseModel.build_pair_siamese_model(base_model)
            
            # Compile with Contrastive Loss
            loss_fn = ContrastiveLoss(margin=settings.SIAMESE_MARGIN)
            
        elif loss_name == "triplet":
            training_logger.info("Setting up Triplet Loss Siamese training...")
            
            # Generate triplets
            triplets = generate_triplets(df_valid)
            
            # Split train/val (80/20)
            train_triplets, val_triplets = train_test_split(
                triplets, test_size=0.2, random_state=settings.RANDOM_SEED
            )

            # Build tf.data datasets
            train_ds = SiameseDatasetBuilder.build_triplet_dataset(train_triplets, is_training=True)
            val_ds = SiameseDatasetBuilder.build_triplet_dataset(val_triplets, is_training=False)

            # Build Siamese model wrapper
            siamese_model = SiameseModel.build_triplet_siamese_model(base_model)
            
            # Compile with Triplet Loss
            loss_fn = TripletLoss(margin=settings.SIAMESE_MARGIN)
            
        else:
            msg = f"Unsupported Siamese loss function: {loss_name}"
            training_logger.error(msg)
            raise ValueError(msg)

        # Compile Keras Siamese Model
        optimizer = tf.keras.optimizers.Adam(learning_rate=settings.SIAMESE_LEARNING_RATE)
        siamese_model.compile(
            optimizer=optimizer,
            loss=loss_fn
        )

        # 4. Fit Siamese Network
        temp_checkpoint_dir = self.checkpoints_dir / f"temp_{settings.MODEL_NAME}_siamese"
        callbacks = get_callbacks(temp_checkpoint_dir, monitor_metric="val_loss", patience=5)

        training_logger.info("Starting Siamese training fit loop...")
        t0 = time.perf_counter()
        history = siamese_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=settings.SIAMESE_EPOCHS,
            callbacks=callbacks
        )
        elapsed_time = time.perf_counter() - t0
        training_logger.info(f"Siamese training fit completed in {elapsed_time:.2f} seconds.")

        # 5. Export embedding model checkpoint
        # The base_model instance shares weights with the trained siamese_model, so it now has the updated weights!
        metrics_summary = {
            "val_loss": float(np.min(history.history["val_loss"])),
            "train_time_sec": elapsed_time
        }
        
        self.checkpoint_manager.save_checkpoint(
            model=base_model,
            model_name=f"{settings.MODEL_NAME}_siamese",
            version=version,
            history_data=history.history,
            metrics_summary=metrics_summary
        )

        # Export training curves
        plot_training_curves(history.history, f"{settings.MODEL_NAME}_siamese_curves")

        # Clean up temp checkpoint folder
        try:
            import shutil
            shutil.rmtree(temp_checkpoint_dir, ignore_errors=True)
        except Exception:
            pass

        return base_model, history.history
