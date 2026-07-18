"""
Module Description: Siamese Loss Functions
Purpose: Vectorized Contrastive Loss and Triplet Loss implementations in TensorFlow.
Author: Technical Lead
Version: 1.0.0
Dependencies: tensorflow, config.settings, utils.logger
"""

import tensorflow as tf
from config import settings
from utils.logger import training_logger


class ContrastiveLoss(tf.keras.losses.Loss):
    """
    Implements Contrastive Loss for Siamese pair distance optimization:
    Loss = Mean( y * d^2 + (1 - y) * max(margin - d, 0)^2 )
    where y = 1 for positive pairs (same class) and y = 0 for negative pairs (different class).
    """

    def __init__(self, margin: float = settings.SIAMESE_MARGIN, **kwargs) -> None:
        """
        Initializes Contrastive Loss.

        Args:
            margin: The minimum separation distance between dissimilar samples.
        """
        super().__init__(**kwargs)
        self.margin: float = margin

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the loss value.

        Args:
            y_true: True labels (1 for similar, 0 for dissimilar) of shape (batch_size,).
            y_pred: Euclidean distances between embedding pairs of shape (batch_size,).

        Returns:
            Reduced mean loss tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        
        # Positive pairs: minimize squared distance
        pos_loss = y_true * tf.square(y_pred)
        
        # Negative pairs: push apart beyond margin
        neg_loss = (1.0 - y_true) * tf.square(tf.maximum(self.margin - y_pred, 0.0))
        
        return tf.reduce_mean(pos_loss + neg_loss)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"margin": self.margin})
        return config


class TripletLoss(tf.keras.losses.Loss):
    """
    Implements Triplet Loss:
    Loss = Max( d(A, P)^2 - d(A, N)^2 + margin, 0 )
    where A = Anchor, P = Positive, N = Negative.
    """

    def __init__(self, margin: float = settings.SIAMESE_MARGIN, **kwargs) -> None:
        """
        Initializes Triplet Loss.

        Args:
            margin: The minimum separation distance between positive and negative pairs.
        """
        super().__init__(**kwargs)
        self.margin: float = margin

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the triplet loss.

        Args:
            y_true: Dummy labels (unused but required by Keras loss signature).
            y_pred: Vector containing concatenated distances [d_ap, d_an].

        Returns:
            Reduced mean loss tensor.
        """
        # Split predicted distances
        # y_pred is expected to be of shape (batch_size, 2) where:
        # y_pred[:, 0] is distance(Anchor, Positive)
        # y_pred[:, 1] is distance(Anchor, Negative)
        d_ap = y_pred[:, 0]
        d_an = y_pred[:, 1]
        
        loss = tf.maximum(tf.square(d_ap) - tf.square(d_an) + self.margin, 0.0)
        return tf.reduce_mean(loss)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"margin": self.margin})
        return config
