"""
Module Description: Siamese Dataset Pipeline
Purpose: Creates tf.data.Dataset pipelines for streaming preloaded pairs and triplets of images.
Author: Technical Lead
Version: 1.2.0
Dependencies: tensorflow, typing, config.settings, utils.logger, preprocessing.image_preprocessor
"""

from typing import List, Tuple, Dict
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from config import settings
from utils.logger import training_logger
from preprocessing.image_preprocessor import ImagePreprocessor


class SiameseDatasetBuilder:
    """
    Helper to construct memory-cached, Keras-compatible data pipelines for training Siamese networks.
    """
    _preloaded_cache: Dict[str, np.ndarray] = {}

    @classmethod
    def preload_all_images(cls, all_paths: List[str]) -> None:
        """
        Loads all subset images into RAM to eliminate disk I/O bottlenecking during training.

        Args:
            all_paths: List of relative or absolute image path strings.
        """
        training_logger.info(f"Preloading {len(all_paths)} images into RAM cache for Siamese training...")
        cls._preloaded_cache = {}
        
        for idx, path_str in enumerate(all_paths):
            try:
                p = Path(path_str)
                if not p.is_absolute():
                    p = settings.BASE_DIR / p
                
                # Use ImagePreprocessor to preprocess to normalized float tensor
                tensor = ImagePreprocessor.preprocess_image(p)
                if tensor is not None:
                    # Store as numpy array
                    cls._preloaded_cache[path_str] = tensor.numpy()
                
                if (idx + 1) % 400 == 0 or idx == len(all_paths) - 1:
                    training_logger.info(f"Preloaded {idx + 1} / {len(all_paths)} images.")
            except Exception as e:
                training_logger.warning(f"Failed to preload image {path_str}: {e}")
                
        training_logger.info("Image preloading completed successfully.")

    @classmethod
    def build_pair_dataset(cls, 
                           pairs: List[Tuple[str, str]], 
                           labels: List[int], 
                           batch_size: int = settings.BATCH_SIZE,
                           is_training: bool = True) -> tf.data.Dataset:
        """
        Builds a tf.data.Dataset for pair-based Contrastive training from RAM-cached images.
        """
        training_logger.info(f"Building pair dataset (size: {len(pairs)}, training={is_training})...")

        X1_list = []
        X2_list = []
        y_list = []

        for (p1, p2), label in zip(pairs, labels):
            if p1 in cls._preloaded_cache and p2 in cls._preloaded_cache:
                X1_list.append(cls._preloaded_cache[p1])
                X2_list.append(cls._preloaded_cache[p2])
                y_list.append(label)

        if not X1_list:
            raise ValueError("No preloaded images matched the generated pair paths. Ensure preload_all_images is called.")

        X1 = np.array(X1_list, dtype=np.float32)
        X2 = np.array(X2_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        ds = tf.data.Dataset.from_tensor_slices(((X1, X2), y))
        
        if is_training:
            ds = ds.shuffle(buffer_size=1000, seed=settings.RANDOM_SEED)
            
            # Apply Keras data augmentations to pairs on-the-fly
            from preprocessing.augmentation import DataAugmentor
            augmentor = DataAugmentor(enabled=True)
            aug_layer = augmentor.get_augmentation_layer()
            ds = ds.map(
                lambda imgs, label: ((aug_layer(imgs[0], training=True), aug_layer(imgs[1], training=True)), label),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    @classmethod
    def build_triplet_dataset(cls, 
                              triplets: List[Tuple[str, str, str]], 
                              batch_size: int = settings.BATCH_SIZE,
                              is_training: bool = True) -> tf.data.Dataset:
        """
        Builds a tf.data.Dataset for triplet-based training from RAM-cached images.
        """
        training_logger.info(f"Building triplet dataset (size: {len(triplets)}, training={is_training})...")

        anc_list = []
        pos_list = []
        neg_list = []

        for a, p, n in triplets:
            if a in cls._preloaded_cache and p in cls._preloaded_cache and n in cls._preloaded_cache:
                anc_list.append(cls._preloaded_cache[a])
                pos_list.append(cls._preloaded_cache[p])
                neg_list.append(cls._preloaded_cache[n])

        if not anc_list:
            raise ValueError("No preloaded images matched the generated triplet paths.")

        anc = np.array(anc_list, dtype=np.float32)
        pos = np.array(pos_list, dtype=np.float32)
        neg = np.array(neg_list, dtype=np.float32)
        dummy_labels = np.zeros(len(anc), dtype=np.float32)

        ds = tf.data.Dataset.from_tensor_slices(((anc, pos, neg), dummy_labels))
        
        if is_training:
            ds = ds.shuffle(buffer_size=1000, seed=settings.RANDOM_SEED)
            
            from preprocessing.augmentation import DataAugmentor
            augmentor = DataAugmentor(enabled=True)
            aug_layer = augmentor.get_augmentation_layer()
            ds = ds.map(
                lambda imgs, label: (
                    (aug_layer(imgs[0], training=True), aug_layer(imgs[1], training=True), aug_layer(imgs[2], training=True)),
                    label
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds
