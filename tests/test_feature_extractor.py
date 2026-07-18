"""
Module Description: Feature Extractor Unit Tests
Purpose: Assures backbones and custom projection layers are built with correct output shapes and frozen settings.
Author: Technical Lead
Version: 1.0.0
Dependencies: pytest, tensorflow, models.model_factory, models.embedding_head
"""

import pytest
import tensorflow as tf
from models.model_factory import ModelFactory
from models.embedding_head import EmbeddingHead


def test_embedding_head_shape() -> None:
    """Ensures custom projection layer outputs L2 normalized vectors of target dimension."""
    emb_dim = 128
    layer = EmbeddingHead(embedding_dim=emb_dim, dropout_rate=0.2)
    
    # Random input tensor: shape (batch_size, height, width, channels)
    dummy_input = tf.random.normal((2, 7, 7, 256))
    output = layer(dummy_input, training=False)
    
    assert output.shape == (2, emb_dim)
    
    # Verify L2 normalization: row norms should be equal to 1.0
    norms = tf.norm(output, axis=1).numpy()
    assert pytest.approx(norms[0]) == 1.0
    assert pytest.approx(norms[1]) == 1.0


def test_model_factory_backbone() -> None:
    """Ensures backbones are loaded and frozen correctly."""
    # We can try loading a small model or mock it to save memory and time
    # Let's verify factory functions return functional models
    model = ModelFactory.build_embedding_model(
        model_name="resnet50",
        embedding_dim=256,
        freeze_backbone=True
    )
    
    assert isinstance(model, tf.keras.Model)
    assert model.output_shape == (None, 256)
    
    # Check that layers of backbone are frozen (trainable = False)
    backbone_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone_layer = layer
            break
            
    if backbone_layer:
        assert not backbone_layer.trainable
