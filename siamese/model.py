"""
Module Description: Siamese Network Architecture
Purpose: Implements the weight-sharing Siamese model for pair-based and triplet-based metric learning.
Author: Technical Lead
Version: 1.0.0
Dependencies: tensorflow, config.settings, utils.logger, models.model_factory
"""

from typing import Tuple
import tensorflow as tf
from config import settings
from utils.logger import training_logger


class DistanceLayer(tf.keras.layers.Layer):
    """
    A Keras layer to compute the Euclidean distance between two embedding vectors.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Computes Euclidean distance: d = sqrt( sum( (x1 - x2)^2, axis=1 ) )

        Args:
            inputs: A tuple of two tensors (emb1, emb2) each of shape (batch_size, embedding_dim).

        Returns:
            Distance tensor of shape (batch_size,).
        """
        emb1, emb2 = inputs
        # Small epsilon to prevent NaN gradients on zero distance
        squared_difference = tf.reduce_sum(tf.square(emb1 - emb2), axis=1)
        return tf.sqrt(tf.maximum(squared_difference, 1e-7))


class SiameseModel:
    """
    Constructs the shared-weight Siamese model for pair or triplet training.
    """

    @staticmethod
    def build_pair_siamese_model(base_embedding_model: tf.keras.Model) -> tf.keras.Model:
        """
        Builds a Siamese network that takes two image inputs and outputs their Euclidean distance.

        Args:
            base_embedding_model: The shared feature extraction model (Backbone + EmbeddingHead).

        Returns:
            tf.keras.Model representing the Siamese Pair Network.
        """
        training_logger.info("Building Siamese Pair Network with shared weights...")

        # Inputs
        input_image_1 = tf.keras.Input(shape=(*settings.IMAGE_SIZE, 3), name="input_image_1")
        input_image_2 = tf.keras.Input(shape=(*settings.IMAGE_SIZE, 3), name="input_image_2")

        # Shared forward pass (extract embeddings)
        emb_1 = base_embedding_model(input_image_1)
        emb_2 = base_embedding_model(input_image_2)

        # Distance computation layer
        distance = DistanceLayer(name="distance_layer")((emb_1, emb_2))

        # Functional model
        model = tf.keras.Model(
            inputs=[input_image_1, input_image_2], 
            outputs=distance, 
            name="siamese_pair_model"
        )
        training_logger.info("Siamese Pair Network built successfully.")
        return model

    @staticmethod
    def build_triplet_siamese_model(base_embedding_model: tf.keras.Model) -> tf.keras.Model:
        """
        Builds a Siamese network that takes three image inputs (Anchor, Positive, Negative)
        and outputs a concatenated tensor of [dist_ap, dist_an].

        Args:
            base_embedding_model: The shared feature extraction model.

        Returns:
            tf.keras.Model representing the Siamese Triplet Network.
        """
        training_logger.info("Building Siamese Triplet Network with shared weights...")

        # Inputs
        input_anchor = tf.keras.Input(shape=(*settings.IMAGE_SIZE, 3), name="input_anchor")
        input_positive = tf.keras.Input(shape=(*settings.IMAGE_SIZE, 3), name="input_positive")
        input_negative = tf.keras.Input(shape=(*settings.IMAGE_SIZE, 3), name="input_negative")

        # Shared forward pass
        emb_anchor = base_embedding_model(input_anchor)
        emb_positive = base_embedding_model(input_positive)
        emb_negative = base_embedding_model(input_negative)

        # Distance layers
        distance_ap = DistanceLayer(name="dist_ap")((emb_anchor, emb_positive))
        distance_an = DistanceLayer(name="dist_an")((emb_anchor, emb_negative))

        # Stack distances into shape (batch_size, 2)
        # Output columns: [dist_ap, dist_an]
        distances = tf.keras.layers.concatenate([distance_ap[..., tf.newaxis], distance_an[..., tf.newaxis]], axis=1, name="distance_stack")

        model = tf.keras.Model(
            inputs=[input_anchor, input_positive, input_negative], 
            outputs=distances, 
            name="siamese_triplet_model"
        )
        training_logger.info("Siamese Triplet Network built successfully.")
        return model
