"""
Module Description: Upload Widget Component
Purpose: Renders the file upload dialog and handles validation and preview.
Author: Technical Lead
Version: 1.0.0
Dependencies: streamlit, PIL.Image, config.settings, utils.logger
"""

from typing import Tuple, Optional
import streamlit as st
from PIL import Image
from config import settings
from utils.logger import app_logger
from preprocessing.image_validator import ImageValidator


def render_upload_widget() -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Draws image uploader card and performs extensions checks.

    Returns:
        A tuple of (loaded_pil_image, image_name) or (None, None).
    """
    st.markdown("### 📤 Upload Fashion Product Image")
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, JPEG, PNG)...",
        type=["jpg", "jpeg", "png"],
        key="file_uploader_widget"
    )

    if uploaded_file is not None:
        try:
            # Load PIL Image
            img = Image.open(uploaded_file)
            
            # Temporary file write check
            temp_path = settings.SUBSET_DIRECTORY / f"temp_upload_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Audit file constraints
            is_valid, error_msg = ImageValidator.validate_image_file(temp_path)
            if not is_valid:
                st.error(f"Invalid image file: {error_msg}")
                # Clean up
                if temp_path.exists():
                    temp_path.unlink()
                return None, None

            # Convert to RGB to ensure uniform representation
            img_rgb = img.convert("RGB")

            # Clean up temp upload file
            if temp_path.exists():
                temp_path.unlink()

            st.success(f"Image '{uploaded_file.name}' uploaded successfully.")
            
            # Render side-by-side layout for preview
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(img_rgb, caption="Query Preview", use_column_width=True)
            with col2:
                st.info(
                    f"**Details**:\n"
                    f"- **Dimensions**: {img_rgb.size[0]} x {img_rgb.size[1]} pixels\n"
                    f"- **Format**: {uploaded_file.type}\n"
                    f"- **Size**: {len(uploaded_file.getvalue()) / 1024:.2f} KB"
                )

            return img_rgb, uploaded_file.name

        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            app_logger.error(f"Uploader error: {e}")
            return None, None

    return None, None
