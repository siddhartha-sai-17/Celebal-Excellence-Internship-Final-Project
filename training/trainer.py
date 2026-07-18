"""
Module Description: Transfer Learning Trainer
Purpose: Manages two-stage transfer learning: Stage 1 (head training) and Stage 2 (fine-tuning backbone) on category labels.
Author: Technical Lead
Version: 1.0.0
Dependencies: pandas, numpy, tensorflow, sklearn, config.settings, utils.logger, models.model_factory, models.checkpoint_manager, training.callbacks, training.history_logger
"""

import time
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from config import settings
from utils.logger import training_logger
from utils.timer import time_function
from models.model_factory import ModelFactory
from models.checkpoint_manager import CheckpointManager
from models.model_utils import configure_accelerator
from training.callbacks import get_callbacks
from training.history_logger import HistoryLogger
from utils.visualization import plot_training_curves


class TransferLearningTrainer:
    """
    Orchestrates category classifier training, weight checkpoints, and validation updates.
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

    def _load_and_prepare_dataset(self) -> Tuple[List[str], List[int], LabelEncoder]:
        """
        Loads subset metadata and encodes category string labels to integers.

        Returns:
            A tuple of (image_paths, encoded_labels, label_encoder).
        """
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
        training_logger.info(f"Loaded {len(df_valid)} valid subset image rows.")

        image_paths = df_valid["image_path"].tolist()
        categories = df_valid["articleType"].tolist()

        # Label encoding
        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(categories).tolist()

        # Log class mapping
        class_mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
        training_logger.info(f"Class mapping: {class_mapping}")

        return image_paths, encoded_labels, encoder

    @staticmethod
    def _parse_image_fn(img_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        TensorFlow dataset mapping function: reads, decodes, resizes, and normalizes.
        """
        # Read image file
        img_bytes = tf.io.read_file(img_path)
        # Decode jpeg
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        # Resize using bilinear interpolation
        img = tf.image.resize(img, settings.IMAGE_SIZE, method="bilinear")
        # Normalize in range [0, 1] then map using ImageNet stats
        img = tf.cast(img, tf.float32) / 255.0
        
        mean = tf.constant(settings.IMAGENET_MEAN, dtype=tf.float32)
        std = tf.constant(settings.IMAGENET_STD, dtype=tf.float32)
        img_normalized = (img - mean) / std

        return img_normalized, label

    def _create_tf_datasets(self, 
                            train_paths: List[str], 
                            train_labels: List[int],
                            val_paths: List[str], 
                            val_labels: List[int],
                            num_classes: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Builds optimized training and validation tf.data.Dataset pipelines.
        """
        # 1. Training dataset
        train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        # Map preprocessing
        train_ds = train_ds.map(self._parse_image_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Apply data augmentation if configured
        from preprocessing.augmentation import DataAugmentor
        augmentor = DataAugmentor(enabled=True)
        aug_layer = augmentor.get_augmentation_layer()
        
        # Augment maps batch
        train_ds = train_ds.shuffle(buffer_size=1000, seed=settings.RANDOM_SEED)
        train_ds = train_ds.batch(settings.BATCH_SIZE)
        
        # Apply augmentation function after batching for vectorized acceleration
        train_ds = train_ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        # 2. Validation dataset
        val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        val_ds = val_ds.map(self._parse_image_fn, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(settings.BATCH_SIZE)
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, val_ds

    @time_function("transfer_learning_training")
    def run_training(self, version: str = "v1") -> Tuple[tf.keras.Model, Dict[str, Any]]:
        """
        Executes two-stage transfer learning:
        Stage 1: Trains frozen backbone classification head.
        Stage 2: Unfreezes backbone last layers and fine-tunes.

        Args:
            version: Target version identifier.

        Returns:
            A tuple of (trained_embedding_model, history_metrics).
        """
        training_logger.info("Initializing Transfer Learning trainer...")
        
        # Load dataset
        paths, labels, encoder = self._load_and_prepare_dataset()
        num_classes = len(encoder.classes_)

        # Split train/val sets (80/20)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            paths, labels, test_size=0.2, random_state=settings.RANDOM_SEED, stratify=labels
        )

        training_logger.info(f"Split sizes: Train: {len(train_paths)}, Val: {len(val_paths)}")

        # Create tf.data datasets
        train_ds, val_ds = self._create_tf_datasets(
            train_paths, train_labels, val_paths, val_labels, num_classes
        )

        # ----------------------------------------------------------------------
        # STAGE 1: Train Classification Head (Backbone Frozen)
        # ----------------------------------------------------------------------
        training_logger.info("--- STAGE 1: Training Classification Head (Backbone Frozen) ---")
        
        # Build classification model
        model = ModelFactory.build_classification_model(
            num_classes=num_classes,
            model_name=settings.MODEL_NAME,
            dropout_rate=settings.DROPOUT_RATE,
            freeze_backbone=True
        )

        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=settings.LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Temp directory for stage 1 checkpoints
        stage1_dir = self.checkpoints_dir / f"temp_{settings.MODEL_NAME}_stage1"
        stage1_callbacks = get_callbacks(stage1_dir, monitor_metric="val_loss")

        # Fit Stage 1
        t0 = time.perf_counter()
        history_stage1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=settings.EPOCHS,
            callbacks=stage1_callbacks
        )
        stage1_time = time.perf_counter() - t0
        training_logger.info(f"Stage 1 completed in {stage1_time:.2f} seconds.")

        # ----------------------------------------------------------------------
        # STAGE 2: Fine-Tuning (Unfreeze Last Blocks)
        # ----------------------------------------------------------------------
        training_logger.info("--- STAGE 2: Fine-Tuning Last Backbone Blocks ---")
        
        # Unfreeze last layers (e.g. 15 layers)
        ModelFactory.unfreeze_backbone_layers(model, num_layers_to_unfreeze=20)

        # Compile model with a very small learning rate for fine tuning
        fine_tune_optimizer = tf.keras.optimizers.Adam(learning_rate=settings.FINE_TUNE_LEARNING_RATE)
        model.compile(
            optimizer=fine_tune_optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Fit Stage 2
        stage2_dir = self.checkpoints_dir / f"temp_{settings.MODEL_NAME}_stage2"
        stage2_callbacks = get_callbacks(stage2_dir, monitor_metric="val_loss")

        t0 = time.perf_counter()
        history_stage2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=settings.FINE_TUNE_EPOCHS,
            callbacks=stage2_callbacks
        )
        stage2_time = time.perf_counter() - t0
        training_logger.info(f"Stage 2 fine-tuning completed in {stage2_time:.2f} seconds.")

        # ----------------------------------------------------------------------
        # POST-TRAINING: Save model weights and build embedding model
        # ----------------------------------------------------------------------
        # Combine historical logs
        combined_history = {}
        for key in history_stage1.history.keys():
            combined_history[key] = history_stage1.history[key] + history_stage2.history[key]

        # Build embedding model representation (Backbone + EmbeddingHead)
        embedding_model = ModelFactory.build_embedding_model(
            model_name=settings.MODEL_NAME,
            embedding_dim=settings.EMBEDDING_DIMENSION,
            dropout_rate=settings.DROPOUT_RATE,
            freeze_backbone=False
        )

        # Copy backbone weights from classification model to embedding model
        # To do this safely: we can retrieve layers by name or copy weights of the backbone
        # In functional API, the backbone model is the second layer (index 1)
        # Let's extract backbone from classification model
        backbone_class = None
        for l in model.layers:
            if isinstance(l, tf.keras.Model):
                backbone_class = l
                break

        # Find backbone in embedding model
        backbone_emb = None
        for l in embedding_model.layers:
            if isinstance(l, tf.keras.Model):
                backbone_emb = l
                break

        if backbone_class and backbone_emb:
            training_logger.info("Transferring backbone weights to Embedding model...")
            backbone_emb.set_weights(backbone_class.get_weights())
        else:
            training_logger.warning("Could not locate nested backbone sub-models. Weights not synced.")

        # Save embedding model checkpoint
        metrics_summary = {
            "val_loss": float(np.min(combined_history["val_loss"])),
            "val_accuracy": float(np.max(combined_history["val_accuracy"])),
            "train_time_sec": stage1_time + stage2_time
        }
        
        save_dir = self.checkpoint_manager.save_checkpoint(
            model=embedding_model,
            model_name=f"{settings.MODEL_NAME}_transfer",
            version=version,
            history_data=combined_history,
            metrics_summary=metrics_summary
        )

        # Save label encoder categories for mapping classes
        np.save(str(save_dir / "classes.npy"), encoder.classes_)

        # Export training charts
        plot_training_curves(combined_history, f"{settings.MODEL_NAME}_transfer_curves")

        # Clean up temp folders
        try:
            import shutil
            shutil.rmtree(stage1_dir, ignore_errors=True)
            shutil.rmtree(stage2_dir, ignore_errors=True)
        except Exception:
            pass

        return embedding_model, combined_history
