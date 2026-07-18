"""
Module Description: Image and Explainability Utilities
Purpose: Safe PIL/numpy conversions and Grad-CAM implementation to visualize model focus.
Author: Technical Lead
Version: 1.0.0
Dependencies: tensorflow, numpy, cv2, matplotlib.cm, config.settings, utils.logger
"""

from pathlib import Path
from typing import Union, Tuple, Optional
import cv2
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from PIL import Image
from config import settings
from utils.logger import app_logger


def resize_image_with_aspect_ratio(img: Image.Image, target_size: Tuple[int, int] = settings.IMAGE_SIZE) -> Image.Image:
    """
    Resizes a PIL image maintaining aspect ratio and pads it to target size.

    Args:
        img: Original PIL Image.
        target_size: Target size tuple (width, height).

    Returns:
        Padded and resized PIL Image.
    """
    img.thumbnail(target_size, Image.Resampling.BILINEAR)
    # Create black background image of target size
    background = Image.new("RGB", target_size, (0, 0, 0))
    # Paste resized image in the center
    offset = ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2)
    background.paste(img, offset)
    return background


def generate_gradcam(model: tf.keras.Model, 
                     img_tensor: tf.Tensor, 
                     last_conv_layer_name: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Computes a Grad-CAM activation heatmap for the last convolutional layer of the model.

    Args:
        model: Keras Model (embedding model or classification model).
        img_tensor: Preprocessed image tensor with shape (1, 224, 224, 3).
        last_conv_layer_name: Explicit name of the last conv layer. If None, auto-detected.

    Returns:
        2D numpy array heatmap of shape (224, 224), normalized to [0, 1], or None if computation failed.
    """
    try:
        # Find the backbone model inside the functional wrapper
        backbone = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                backbone = layer
                break
        
        target_model = backbone if backbone is not None else model

        # Auto-detect last convolutional layer if not specified
        if last_conv_layer_name is None:
            for layer in reversed(target_model.layers):
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)) or "conv" in layer.name:
                    last_conv_layer_name = layer.name
                    break
        
        if last_conv_layer_name is None:
            app_logger.warning("Could not auto-detect any convolutional layer in the model.")
            return None

        app_logger.info(f"Computing Grad-CAM using layer '{last_conv_layer_name}'...")

        # Create a model that maps the inputs to both the activations of the last conv layer and the final output
        grad_model = tf.keras.models.Model(
            inputs=[target_model.input],
            outputs=[target_model.get_layer(last_conv_layer_name).output, target_model.output]
        )

        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # If the model has nested layers, ensure correct input flow
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            
            # Since this is an embedding model, the output is a 512-dim embedding.
            # We compute gradients with respect to the L2-norm of the embedding (or the maximum prediction value)
            if len(predictions.shape) == 2 and predictions.shape[1] > 1:
                # For classification, use top predicted class logit
                class_idx = tf.argmax(predictions[0])
                loss_value = predictions[:, class_idx]
            else:
                # For embedding model, maximize the L2-norm or sum of channels
                loss_value = tf.reduce_sum(predictions)

        # Get gradients of the loss with respect to output feature map of conv layer
        grads = tape.gradient(loss_value, conv_outputs)

        # Compute guided gradients (pool over channels)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Multiply feature map by pooled gradients to weight important channels
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU to keep only positive contributions and normalize to [0, 1]
        heatmap = tf.maximum(heatmap, 0.0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0.0:
            heatmap = heatmap / max_val

        return heatmap.numpy()

    except Exception as e:
        app_logger.error(f"Grad-CAM generation failed: {e}")
        return None


def overlay_heatmap_on_image(image_path: Union[str, Path], heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    Overlays a Grad-CAM heatmap on top of the original image.

    Args:
        image_path: Path to original image.
        heatmap: 2D numpy array heatmap of normalized values in range [0, 1].
        alpha: Transparency factor of heatmap overlay.

    Returns:
        PIL Image showing original image superimposed with the heatmap.
    """
    try:
        # Load original image with OpenCV
        img = cv2.imread(str(image_path))
        img = cv2.resize(img, settings.IMAGE_SIZE)

        # Rescale heatmap to [0, 255]
        heatmap_255 = np.uint8(255 * heatmap)

        # Use colormaps interface safely
        import matplotlib
        if hasattr(matplotlib, "colormaps"):
            jet = matplotlib.colormaps["jet"]
        else:
            jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_colors = np.uint8(255 * jet_colors)
        
        # Color map mapping
        colored_heatmap = jet_colors[heatmap_255]
        colored_heatmap = cv2.resize(colored_heatmap, (img.shape[1], img.shape[0]))

        # Superimpose
        superimposed = colored_heatmap * alpha + img * (1.0 - alpha)
        superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

        # Convert back to RGB for PIL Image display
        superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
        return Image.fromarray(superimposed_rgb)
    except Exception as e:
        app_logger.error(f"Overlaying heatmap failed: {e}")
        # Return fallback original image resized
        return Image.open(image_path).resize(settings.IMAGE_SIZE)
