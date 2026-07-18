"""
Module Description: Safe Image Loader
Purpose: Loads images safely into memory, converts grayscale/RGBA to standard RGB mode, and handles failures.
Author: Technical Lead
Version: 1.0.0
Dependencies: PIL.Image, config.settings, utils.logger, preprocessing.image_validator
"""

from pathlib import Path
from typing import Optional, Union
from PIL import Image
from config import settings
from utils.logger import app_logger
from preprocessing.image_validator import ImageValidator


class ImageLoader:
    """
    Responsible for loading image files safely, handling formats, and converting to standard RGB mode.
    """

    @staticmethod
    def load_image(image_path: Union[str, Path]) -> Optional[Image.Image]:
        """
        Loads an image file safely, converting it to RGB mode.

        Args:
            image_path: Path to the image file to load.

        Returns:
            PIL Image object in RGB mode, or None if validation or loading failed.
        """
        path = Path(image_path)
        
        # Validate the image file first
        is_valid, error_msg = ImageValidator.validate_image_file(path)
        if not is_valid:
            app_logger.warning(f"Skipping loading of {path} due to validation failure: {error_msg}")
            return None

        try:
            with Image.open(path) as img:
                # Load image content into memory and close file pointer
                img_rgb = img.convert("RGB")
                img_rgb.load()
                app_logger.info(f"Loaded image successfully: {path} (original mode: {img.mode}, dimensions: {img.size})")
                return img_rgb
        except Exception as e:
            app_logger.error(f"Failed to load image from {path} despite validation pass: {e}")
            return None
