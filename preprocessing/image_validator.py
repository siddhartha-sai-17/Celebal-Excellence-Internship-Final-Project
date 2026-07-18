"""
Module Description: Image Validation Module
Purpose: Validates image files for formatting, dimension correctness, file size, corruption, and readability.
Author: Technical Lead
Version: 1.0.0
Dependencies: PIL.Image, config.settings, utils.logger
"""

import os
from pathlib import Path
from typing import Tuple, Union
from PIL import Image
from config import settings
from utils.logger import app_logger


class ImageValidator:
    """
    Validates images using format checks, dimension checks, file sizes, and corruption checks.
    """

    @staticmethod
    def validate_image_file(image_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Validates an image file's existence, size, extension, format, and corruption.

        Args:
            image_path: Path to the image file to validate.

        Returns:
            A tuple of (is_valid, error_message).
        """
        path = Path(image_path)

        # Check existence
        if not path.exists():
            msg = f"File does not exist: {path}"
            app_logger.warning(msg)
            return False, msg

        # Check file size (should not be empty or too large)
        try:
            file_size_bytes = path.stat().st_size
            if file_size_bytes == 0:
                msg = f"Empty file: {path}"
                app_logger.warning(msg)
                return False, msg

            file_size_mb = file_size_bytes / (1024 * 1024)
            if file_size_mb > settings.MAX_IMAGE_FILE_SIZE_MB:
                msg = f"File size too large: {file_size_mb:.2f}MB (Max allowed: {settings.MAX_IMAGE_FILE_SIZE_MB}MB)"
                app_logger.warning(msg)
                return False, msg
        except OSError as e:
            msg = f"Unable to read file statistics for {path}: {e}"
            app_logger.error(msg)
            return False, msg

        # Check extension
        extension = path.suffix.lower()
        if extension not in settings.SUPPORTED_IMAGE_FORMATS:
            msg = f"Unsupported file extension '{extension}' for file {path}. Allowed: {settings.SUPPORTED_IMAGE_FORMATS}"
            app_logger.warning(msg)
            return False, msg

        # Verify image readable and not corrupted
        try:
            with Image.open(path) as img:
                # Force PIL to load the entire image content to detect corruption
                img.verify()
        except (IOError, SyntaxError) as e:
            msg = f"Corrupted or invalid image content for file {path}: {e}"
            app_logger.warning(msg)
            return False, msg

        # Try to read dimensions and mode
        try:
            with Image.open(path) as img:
                width, height = img.size
                mode = img.mode

                # Reject image if dimensions are too small (e.g. 1x1 placeholder)
                if width < 10 or height < 10:
                    msg = f"Invalid image dimensions: {width}x{height} for file {path}"
                    app_logger.warning(msg)
                    return False, msg
                
                # Check for single-channel (grayscale) images
                if mode not in ("RGB", "RGBA", "L"):
                    msg = f"Unsupported image color mode '{mode}' for file {path}"
                    app_logger.warning(msg)
                    return False, msg
        except Exception as e:
            msg = f"Error reading image details for file {path}: {e}"
            app_logger.error(msg)
            return False, msg

        return True, ""
