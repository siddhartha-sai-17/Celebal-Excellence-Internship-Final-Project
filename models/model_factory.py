"""
Module Description: Model Architecture Factory
Purpose: Instantiates and configures baseline, classification, and Siamese network architectures.
Author: Technical Lead
Version: 1.0.0
Dependencies: tensorflow, config.settings, utils.logger, models.resnet_model, models.efficientnet_model, models.embedding_head
"""

import tensorflow as tf
from config import settings
from utils.logger import app_logger
from models.resnet_model import get_resnet_backbone
from models.efficientnet_model import get_efficientnet_backbone
from models.embedding_head import EmbeddingHead, ClassificationHead


class ModelFactory:
    """
    Factory class to instantiate various neural architectures for baseline extraction, classification training, and similarity retrieval.
    """

    @staticmethod
    def get_backbone(model_name: str = settings.MODEL_NAME, freeze: bool = True) -> tf.keras.Model:
        """
        Instantiates the requested backbone model.

        Args:
            model_name: Type of backbone ("resnet50" or "efficientnetb0").
            freeze: If True, freezes backbone layers.

        Returns:
            tf.keras.Model containing the backbone.
        """
        name_lower = model_name.lower()
        if "resnet" in name_lower:
            return get_resnet_backbone(freeze_backbone=freeze)
        elif "efficientnet" in name_lower:
            return get_efficientnet_backbone(freeze_backbone=freeze)
        else:
            msg = f"Unsupported backbone model type: {model_name}"
            app_logger.error(msg)
            raise ValueError(msg)

    @staticmethod
    def build_embedding_model(model_name: str = settings.MODEL_NAME,
                              embedding_dim: int = settings.EMBEDDING_DIMENSION,
                              dropout_rate: float = settings.DROPOUT_RATE,
                              freeze_backbone: bool = True,
                              is_baseline: bool = False) -> tf.keras.Model:
        """
        Builds the standard embedding model.
        For baseline models (is_baseline=True), outputs L2-normalized Global Average Pooled backbone features directly.
        For trained models, outputs projection through EmbeddingHead.
        """
        app_logger.info(f"Building Embedding Model using backbone: {model_name} (is_baseline={is_baseline})...")
        backbone = ModelFactory.get_backbone(model_name, freeze=freeze_backbone)
        
        # Build Keras Functional model
        inputs = tf.keras.Input(shape=(*settings.IMAGE_SIZE, 3), name="image_input")
        features = backbone(inputs)
        
        if is_baseline:
            # Baseline zero-shot CNN uses backbone features directly
            gap = tf.keras.layers.GlobalAveragePooling2D(name="baseline_gap")(features)
            embeddings = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), name="baseline_l2")(gap)
        else:
            # Trained models use the projection EmbeddingHead
            embeddings = EmbeddingHead(embedding_dim=embedding_dim, dropout_rate=dropout_rate, name="embedding_head")(features)
        
        model = tf.keras.Model(inputs=inputs, outputs=embeddings, name=f"{model_name}_embedding_model")
        app_logger.info("Embedding Model built successfully.")
        return model

    @staticmethod
    def build_classification_model(num_classes: int,
                                   model_name: str = settings.MODEL_NAME,
                                   dropout_rate: float = settings.DROPOUT_RATE,
                                   freeze_backbone: bool = True) -> tf.keras.Model:
        """
        Builds the classification training model (Backbone + ClassificationHead).

        Args:
            num_classes: Number of product categories to classify.
            model_name: Type of backbone.
            dropout_rate: Dropout rate.
            freeze_backbone: If True, freezes backbone weights.

        Returns:
            tf.keras.Model containing the classification pipeline.
        """
        app_logger.info(f"Building Classification Model using backbone: {model_name} for {num_classes} classes...")
        backbone = ModelFactory.get_backbone(model_name, freeze=freeze_backbone)

        inputs = tf.keras.Input(shape=(*settings.IMAGE_SIZE, 3), name="image_input")
        features = backbone(inputs)
        logits = ClassificationHead(num_classes=num_classes, dropout_rate=dropout_rate, name="classification_head")(features)

        model = tf.keras.Model(inputs=inputs, outputs=logits, name=f"{model_name}_classification_model")
        app_logger.info("Classification Model built successfully.")
        return model

    @staticmethod
    def unfreeze_backbone_layers(model: tf.keras.Model, num_layers_to_unfreeze: int = 15) -> None:
        """
        Unfreezes the last N layers of the backbone model inside the functional architecture.

        Args:
            model: The full model (embedding or classification) containing a backbone layer.
            num_layers_to_unfreeze: Number of layers starting from the end of the backbone to make trainable.
        """
        app_logger.info(f"Unfreezing the last {num_layers_to_unfreeze} layers of the backbone...")

        # Find backbone layer in functional model
        backbone_layer = None
        for layer in model.layers:
            # Detect nested model which is our backbone
            if isinstance(layer, tf.keras.Model):
                backbone_layer = layer
                break
        
        if backbone_layer is None:
            app_logger.warning("No nested backbone Model found in the input model. Attempting to unfreeze top-level layers...")
            backbone_layer = model

        # Set backbone as trainable
        backbone_layer.trainable = True
        
        # Freezing/Unfreezing sub-layers
        total_layers = len(backbone_layer.layers)
        unfreeze_start_idx = max(0, total_layers - num_layers_to_unfreeze)

        for i, layer in enumerate(backbone_layer.layers):
            if i >= unfreeze_start_idx:
                # BatchNormalization layer statistics are kept frozen for transfer learning stability
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True
            else:
                layer.trainable = False

        app_logger.info(
            f"Backbone layer trainable state updated. "
            f"Total layers: {total_layers}. Unfrozen layers index >= {unfreeze_start_idx}."
        )
