"""
Module Description: Siamese Loss Unit Tests
Purpose: Validates mathematical values of Contrastive and Triplet Losses in TensorFlow.
Author: Technical Lead
Version: 1.0.0
Dependencies: pytest, tensorflow, siamese.losses
"""

import pytest
import tensorflow as tf
from siamese.losses import ContrastiveLoss, TripletLoss


def test_contrastive_loss() -> None:
    """Validates Contrastive Loss calculations."""
    margin = 1.0
    loss_fn = ContrastiveLoss(margin=margin)
    
    # 1. Positive pair (y = 1): Loss = d^2
    y_true = tf.constant([1.0])
    y_pred = tf.constant([0.4])  # distance d
    loss = loss_fn(y_true, y_pred)
    assert pytest.approx(loss.numpy()) == 0.16  # 0.4^2

    # 2. Negative pair (y = 0) with distance < margin: Loss = (margin - d)^2
    y_true = tf.constant([0.0])
    y_pred = tf.constant([0.4])
    loss = loss_fn(y_true, y_pred)
    assert pytest.approx(loss.numpy()) == 0.36  # (1.0 - 0.4)^2

    # 3. Negative pair (y = 0) with distance >= margin: Loss = 0
    y_true = tf.constant([0.0])
    y_pred = tf.constant([1.5])
    loss = loss_fn(y_true, y_pred)
    assert pytest.approx(loss.numpy()) == 0.0


def test_triplet_loss() -> None:
    """Validates Triplet Loss calculations."""
    margin = 1.0
    loss_fn = TripletLoss(margin=margin)
    
    # Triplet loss expects y_pred to contain [d_ap, d_an]
    # Loss = max(d_ap^2 - d_an^2 + margin, 0)
    
    # Case 1: positive loss
    y_true = tf.constant([0.0])  # dummy
    y_pred = tf.constant([[0.5, 0.8]])  # d_ap = 0.5, d_an = 0.8
    # 0.5^2 - 0.8^2 + 1.0 = 0.25 - 0.64 + 1.0 = 0.61
    loss = loss_fn(y_true, y_pred)
    assert pytest.approx(loss.numpy()) == 0.61
    
    # Case 2: negative/zero loss
    y_pred = tf.constant([[0.2, 1.5]])  # d_ap = 0.2, d_an = 1.5
    # 0.04 - 2.25 + 1.0 = -1.21 => max(-1.21, 0) = 0.0
    loss = loss_fn(y_true, y_pred)
    assert pytest.approx(loss.numpy()) == 0.0
