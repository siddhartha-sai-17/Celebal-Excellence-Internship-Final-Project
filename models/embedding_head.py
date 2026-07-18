"""
Module Description: Model Embedding Head
Purpose: Implements the projection layers that map backbone features to L2-normalized low-dimensional visual embeddings.
Author: Technical Lead
Version: 1.0.0
Dependencies: tensorflow, config.settings, utils.logger
"""

import tensorflow as tf
from config import settings
from utils.logger import app_logger


class EmbeddingHead(tf.keras.layers.Layer):
    """
    A Keras Layer that implements the embedding head projection:
    GlobalAveragePooling2D -> Dense -> BatchNormalization -> Dropout -> Dense -> L2 Normalization.
    """

    def __init__(self, embedding_dim: int = settings.EMBEDDING_DIMENSION, dropout_rate: float = settings.DROPOUT_RATE, **kwargs) -> None:
        """
        Initializes the embedding head layer.

        Args:
            embedding_dim: Dimensionality of the target embedding space (default 512).
            dropout_rate: Dropout rate (default 0.3).
        """
        super().__init__(**kwargs)
        self.embedding_dim: int = embedding_dim
        self.dropout_rate: float = dropout_rate

        # Layers
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name="embedding_gap")
        
        # Intermediate dense projection
        self.dense_1 = tf.keras.layers.Dense(512, activation="relu", name="embedding_dense_1")
        self.bn_1 = tf.keras.layers.BatchNormalization(name="embedding_bn_1")
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate, name="embedding_dropout")
        
        # Embedding bottleneck projection
        self.dense_2 = tf.keras.layers.Dense(self.embedding_dim, name="embedding_dense_2")
        self.bn_2 = tf.keras.layers.BatchNormalization(name="embedding_bn_2")

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Performs the forward pass.

        Args:
            inputs: Feature maps tensor from CNN backbone.
            training: Boolean flag for Dropout / BatchNormalization layers.

        Returns:
            L2 normalized embedding vector of shape (batch_size, embedding_dim).
        """
        x = self.gap(inputs)
        x = self.dense_1(x)
        x = self.bn_1(x, training=training)
        x = self.dropout(x, training=training)
        x = self.dense_2(x)
        x = self.bn_2(x, training=training)
        # Apply L2 normalization to ensure stable cosine similarity
        embeddings = tf.math.l2_normalize(x, axis=1, name="l2_normalize")
        return embeddings

    def get_config(self) -> dict:
        """Returns the config dictionary for layer serialization."""
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "dropout_rate": self.dropout_rate
        })
        return config


class ClassificationHead(tf.keras.layers.Layer):
    """
    A Keras Layer that implements the classification head for Transfer Learning:
    GlobalAveragePooling2D -> BatchNormalization -> Dropout -> Dense(256, relu) -> BatchNormalization -> Dense(num_classes, softmax).
    """

    def __init__(self, num_classes: int, dropout_rate: float = settings.DROPOUT_RATE, **kwargs) -> None:
        """
        Initializes the classification head layer.

        Args:
            num_classes: Number of product categories to classify.
            dropout_rate: Dropout rate (default 0.3).
        """
        super().__init__(**kwargs)
        self.num_classes: int = num_classes
        self.dropout_rate: float = dropout_rate

        self.gap = tf.keras.layers.GlobalAveragePooling2D(name="class_gap")
        self.bn_1 = tf.keras.layers.BatchNormalization(name="class_bn_1")
        self.dropout_1 = tf.keras.layers.Dropout(self.dropout_rate, name="class_dropout_1")
        self.dense_1 = tf.keras.layers.Dense(256, activation="relu", name="class_dense_1")
        self.bn_2 = tf.keras.layers.BatchNormalization(name="class_bn_2")
        self.dense_out = tf.keras.layers.Dense(self.num_classes, activation="softmax", name="class_softmax")

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Performs forward pass."""
        x = self.gap(inputs)
        x = self.bn_1(x, training=training)
        x = self.dropout_1(x, training=training)
        x = self.dense_1(x)
        x = self.bn_2(x, training=training)
        logits = self.dense_out(x)
        return logits

    def get_config(self) -> dict:
        """Returns configuration details."""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate
        })
        return config
