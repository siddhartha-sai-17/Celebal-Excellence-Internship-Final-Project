"""
Module Description: Siamese Accuracy Metrics
Purpose: Implements custom metrics for measuring Siamese validation accuracy in distance thresholds.
Author: Technical Lead
Version: 1.0.0
Dependencies: tensorflow
"""

import tensorflow as tf


class SiameseAccuracy(tf.keras.metrics.Metric):
    """
    A Keras metric to calculate binary classification accuracy based on a distance threshold.
    """

    def __init__(self, threshold: float = 0.5, name: str = "siamese_accuracy", **kwargs) -> None:
        """
        Initializes the metric.

        Args:
            threshold: Distance threshold (<= threshold is considered a positive match).
        """
        super().__init__(name=name, **kwargs)
        self.threshold: float = threshold
        self.correct = self.add_weight(name="correct_predictions", initializer="zeros")
        self.total = self.add_weight(name="total_predictions", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor = None) -> None:
        """
        Updates metric state.

        Args:
            y_true: True binary labels (1 = similar, 0 = dissimilar).
            y_pred: Euclidean distances.
            sample_weight: Optional weights.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Predict 1 (similar) if distance <= threshold, else 0
        predictions = tf.cast(tf.less_equal(y_pred, self.threshold), tf.float32)
        
        # Check matching
        matches = tf.cast(tf.equal(y_true, predictions), tf.float32)
        
        self.correct.assign_add(tf.reduce_sum(matches))
        self.total.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self) -> tf.Tensor:
        """Computes division."""
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_state(self) -> None:
        """Resets accumulators."""
        self.correct.assign(0.0)
        self.total.assign(0.0)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config
