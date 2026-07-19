"""
Module Description: Kaggle Dataset Auto-Downloader for Cloud Deployment
Purpose: Automatically downloads and extracts the Fashion Product Images dataset
         from Kaggle when running on Streamlit Cloud or any environment where the
         raw dataset is not present locally.
Author: Technical Lead
Version: 1.0.0
Dependencies: kaggle, streamlit, zipfile, pathlib
"""

import os
import sys
import zipfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import app_logger


KAGGLE_DATASET_SLUG = "paramaggarwal/fashion-product-images-small"
KAGGLE_DATASET_SLUG_FULL = "paramaggarwal/fashion-product-images-dataset"


def _configure_kaggle_from_secrets() -> bool:
    """
    Reads Kaggle credentials from Streamlit secrets and sets them as
    environment variables so the kaggle library can authenticate.

    Returns:
        True if credentials were successfully configured, False otherwise.
    """
    try:
        import streamlit as st
        username = st.secrets.get("KAGGLE_USERNAME", "")
        key = st.secrets.get("KAGGLE_KEY", "")
        if username and key:
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key
            app_logger.info("Kaggle credentials configured from Streamlit secrets.")
            return True
        else:
            app_logger.warning("KAGGLE_USERNAME or KAGGLE_KEY not found in Streamlit secrets.")
            return False
    except Exception as e:
        app_logger.warning(f"Could not configure Kaggle credentials: {e}")
        return False


def _configure_kaggle_from_env() -> bool:
    """Checks if Kaggle credentials are already set as environment variables."""
    username = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")
    if username and key:
        app_logger.info("Kaggle credentials found in environment variables.")
        return True
    return False


def dataset_is_present(dataset_dir: Path, styles_csv: Path) -> bool:
    """
    Checks if the raw dataset is already present and usable.

    Args:
        dataset_dir: Path to the images directory.
        styles_csv: Path to the styles/metadata CSV file.

    Returns:
        True if dataset looks complete, False if it needs downloading.
    """
    if not dataset_dir.exists() or not styles_csv.exists():
        return False
    image_count = len(list(dataset_dir.glob("*.jpg")))
    return image_count > 100  # At minimum 100 images should be present


def download_fashion_dataset(target_dir: Path) -> bool:
    """
    Downloads the Fashion Product Images (Small) dataset from Kaggle into
    the specified target directory. Extracts the zip and organises files
    into the expected layout:
        target_dir/
            images/
                <id>.jpg ...
            styles.csv

    Args:
        target_dir: Root directory where the dataset will be stored.

    Returns:
        True if the download and extraction succeeded, False otherwise.
    """
    # Try to configure credentials
    if not _configure_kaggle_from_env():
        if not _configure_kaggle_from_secrets():
            app_logger.error(
                "Kaggle credentials not found. Please add KAGGLE_USERNAME and "
                "KAGGLE_KEY to your Streamlit secrets or environment variables."
            )
            return False

    # Set Kaggle config directory to a writable workspace directory before importing kaggle
    os.environ["KAGGLE_CONFIG_DIR"] = str(target_dir.parent)

    try:
        import kaggle  # noqa: F401
        from kaggle.api.kaggle_api_extended import KaggleApiExtended  # type: ignore

        api = KaggleApiExtended()
        api.authenticate()

        target_dir.mkdir(parents=True, exist_ok=True)
        zip_path = target_dir / "fashion-dataset.zip"

        app_logger.info(f"Downloading Kaggle dataset '{KAGGLE_DATASET_SLUG}' to {target_dir} ...")
        api.dataset_download_files(
            dataset=KAGGLE_DATASET_SLUG,
            path=str(target_dir),
            unzip=False,
            quiet=False,
        )

        # Find the downloaded zip file
        zip_files = list(target_dir.glob("*.zip"))
        if not zip_files:
            app_logger.error("No zip archive found after Kaggle download.")
            return False

        zip_path = zip_files[0]
        app_logger.info(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(target_dir))

        # Clean up zip file after extraction
        zip_path.unlink(missing_ok=True)

        # Locate images and CSV — handle nested directories from Kaggle zip
        _normalise_layout(target_dir)

        app_logger.info("Kaggle dataset downloaded and extracted successfully.")
        return True

    except ImportError:
        app_logger.error(
            "The 'kaggle' package is not installed. Add 'kaggle' to requirements.txt."
        )
        return False
    except Exception as e:
        app_logger.error(f"Kaggle dataset download failed: {e}")
        return False


def _normalise_layout(target_dir: Path) -> None:
    """
    Ensures images land at target_dir/images/*.jpg and the CSV lands at
    target_dir/styles.csv regardless of how Kaggle nested its zip contents.
    """
    images_dst = target_dir / "images"
    styles_dst = target_dir / "styles.csv"

    # Search recursively for the images folder and the styles CSV
    for candidate_images in target_dir.rglob("images"):
        if candidate_images.is_dir() and candidate_images != images_dst:
            candidate_images.rename(images_dst)
            app_logger.info(f"Moved images dir: {candidate_images} -> {images_dst}")
            break

    for candidate_csv in target_dir.rglob("styles.csv"):
        if candidate_csv != styles_dst:
            candidate_csv.rename(styles_dst)
            app_logger.info(f"Moved styles CSV: {candidate_csv} -> {styles_dst}")
            break
