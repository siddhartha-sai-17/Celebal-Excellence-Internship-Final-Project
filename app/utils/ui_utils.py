"""
Module Description: Streamlit Presentation Utilities
Purpose: Encodes PIL Images to base64 for direct HTML embedding, handles sizing, and formatting results.
Author: Technical Lead
Version: 1.0.0
Dependencies: base64, io, PIL.Image, utils.logger
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Union, Optional
from PIL import Image
from utils.logger import app_logger


def pil_to_base64(img: Image.Image) -> str:
    """
    Converts a PIL image to a base64 encoded string.

    Args:
        img: PIL Image object.

    Returns:
        Base64 string.
    """
    try:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        app_logger.error(f"Failed to encode PIL image to base64: {e}")
        return ""


def path_to_base64(image_path: Union[str, Path]) -> str:
    """
    Loads an image file from disk and encodes it to a base64 string.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64 string.
    """
    path = Path(image_path)
    if not path.exists():
        return ""
    try:
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        app_logger.error(f"Failed to read image {path} for base64: {e}")
        return ""
