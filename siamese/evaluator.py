"""
Module Description: Siamese Network Evaluator
Purpose: Evaluates Siamese network predictions (distances) on a validation pair set.
Author: Technical Lead
Version: 1.0.0
Dependencies: numpy, sklearn.metrics, tensorflow, config.settings, utils.logger, siamese.dataset
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import tensorflow as tf
from config import settings
from utils.logger import app_logger
from siamese.dataset import SiameseDatasetBuilder


class SiameseEvaluator:
    """
    Computes performance metrics for similarity thresholds.
    """

    @staticmethod
    def evaluate_pairs(siamese_model: tf.keras.Model, 
                       pairs: List[Tuple[str, str]], 
                       labels: List[int],
                       batch_size: int = settings.BATCH_SIZE) -> Dict[str, Any]:
        """
        Predicts distances for validation pairs and evaluates performance.

        Args:
            siamese_model: Trained pair-based Siamese network.
            pairs: List of image pairs.
            labels: Binary labels.
            batch_size: Batch size.

        Returns:
            Dictionary containing metrics (accuracy, f1, auc, optimal_threshold).
        """
        app_logger.info("Evaluating Siamese Pair Model...")
        
        # Preload unique paths in pairs
        unique_paths = set()
        for p1, p2 in pairs:
            unique_paths.add(p1)
            unique_paths.add(p2)
        SiameseDatasetBuilder.preload_all_images(list(unique_paths))
        
        # Build evaluation dataset (without shuffling or data augmentation)
        val_ds = SiameseDatasetBuilder.build_pair_dataset(pairs, labels, batch_size=batch_size, is_training=False)

        # Predict distances
        distances = []
        true_labels = []
        for x, y in val_ds:
            dists = siamese_model(x, training=False)
            distances.extend(dists.numpy().tolist())
            true_labels.extend(y.numpy().tolist())

        distances = np.array(distances)
        true_labels = np.array(true_labels)

        # In metric learning with Contrastive Loss:
        # Smaller distance indicates positive match (similar class), larger indicates negative.
        # Let's find the threshold that yields the best classification accuracy.
        best_threshold = 0.5
        best_accuracy = 0.0
        
        # Grid search over thresholds in range [0.0, 2.0]
        thresholds = np.linspace(0.0, 2.0, 100)
        for t in thresholds:
            # Predict 1 (similar) if distance <= t, else 0 (dissimilar)
            preds = (distances <= t).astype(int)
            acc = accuracy_score(true_labels, preds)
            if acc > best_accuracy:
                best_accuracy = acc
                best_threshold = t

        # Final predictions at optimal threshold
        final_preds = (distances <= best_threshold).astype(int)
        
        metrics = {
            "accuracy": float(best_accuracy),
            "f1_score": float(f1_score(true_labels, final_preds, zero_division=0)),
            "optimal_threshold": float(best_threshold),
            "mean_positive_distance": float(np.mean(distances[true_labels == 1])),
            "mean_negative_distance": float(np.mean(distances[true_labels == 0]))
        }

        try:
            metrics["auc_score"] = float(roc_auc_score(true_labels, -distances))  # negative distances to preserve rank order
        except Exception:
            metrics["auc_score"] = 0.5

        app_logger.info(f"Siamese Evaluation Metrics: {metrics}")
        return metrics
