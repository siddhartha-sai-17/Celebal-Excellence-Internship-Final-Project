"""
Module Description: Project Configuration Settings
Purpose: Centralizes all project constants, directories, model parameters, and global options.
Author: Technical Lead
Version: 1.0.0
Dependencies: pathlib
"""

import os
from pathlib import Path
from typing import List, Tuple

# ==============================================================================
# PROJECT METADATA
# ==============================================================================
PROJECT_NAME: str = "Image-Based Fashion Product Recommendation System"
PROJECT_VERSION: str = "1.0.0"

# ==============================================================================
# PATH CONFIGURATION (using pathlib)
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Raw and processed directories
DATASET_DIRECTORY: Path = BASE_DIR / "dataset"
RAW_DATA_DIRECTORY: Path = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIRECTORY: Path = BASE_DIR / "data" / "processed"
SUBSET_DIRECTORY: Path = BASE_DIR / "data" / "subset"

# Model and embedding directories
MODEL_DIRECTORY: Path = BASE_DIR / "models"
EMBEDDING_DIRECTORY: Path = BASE_DIR / "embeddings"
FAISS_DIRECTORY: Path = BASE_DIR / "faiss"
LOG_DIRECTORY: Path = BASE_DIR / "logs"
EVALUATION_DIRECTORY: Path = BASE_DIR / "evaluation"
PLOTS_DIRECTORY: Path = EVALUATION_DIRECTORY / "plots"
REPORTS_DIRECTORY: Path = EVALUATION_DIRECTORY / "reports"

# Ensure crucial directories are created on initialization
for path in [
    RAW_DATA_DIRECTORY,
    PROCESSED_DATA_DIRECTORY,
    SUBSET_DIRECTORY,
    MODEL_DIRECTORY,
    EMBEDDING_DIRECTORY,
    FAISS_DIRECTORY,
    LOG_DIRECTORY,
    PLOTS_DIRECTORY,
    REPORTS_DIRECTORY,
]:
    path.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# IMAGE & PREPROCESSING STANDARDS
# ==============================================================================
IMAGE_SIZE: Tuple[int, int] = (224, 224)
SUPPORTED_IMAGE_FORMATS: List[str] = [".jpg", ".jpeg", ".png"]
MAX_IMAGE_FILE_SIZE_MB: float = 15.0  # limit image size to 15MB

# ImageNet statistics for normalization
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# ==============================================================================
# SEED AND REPRODUCIBILITY
# ==============================================================================
RANDOM_SEED: int = 42

# ==============================================================================
# DATASET SUBSET SELECTION
# ==============================================================================
# 5-8 Categories for balanced subset
SELECTED_CATEGORIES: List[str] = [
    "Tshirts",
    "Shirts",
    "Casual Shoes",
    "Watches",
    "Handbags",
    "Sandals",
    "Jeans"
]
SAMPLES_PER_CATEGORY: int = 250  # Balanced dataset size target is ~1750 images

# ==============================================================================
# MODEL CONFIGURATIONS
# ==============================================================================
# Backbone choices: "resnet50" or "efficientnetb0"
MODEL_NAME: str = "resnet50" 
EMBEDDING_DIMENSION: int = 512
DROPOUT_RATE: float = 0.3

# ==============================================================================
# TRANSFER LEARNING HYPERPARAMETERS
# ==============================================================================
BATCH_SIZE: int = 32
EPOCHS: int = 2
FINE_TUNE_EPOCHS: int = 1

LEARNING_RATE: float = 1e-3
FINE_TUNE_LEARNING_RATE: float = 1e-5
EARLY_STOPPING_PATIENCE: int = 5

OPTIMIZER: str = "adam"  # Options: "adam", "adamw", "sgd"
LOSS_FUNCTION: str = "categorical_crossentropy"  # Options: "categorical_crossentropy", "sparse_categorical_crossentropy"

# ==============================================================================
# SIAMESE METRIC LEARNING HYPERPARAMETERS
# ==============================================================================
SIAMESE_LOSS_FUNCTION: str = "contrastive"  # Options: "contrastive", "triplet"
SIAMESE_MARGIN: float = 1.0
SIAMESE_EPOCHS: int = 2
SIAMESE_LEARNING_RATE: float = 1e-4
SIAMESE_OPTIMIZER: str = "adam"

# ==============================================================================
# RECOMMENDATION & SIMILARITY SEARCH
# ==============================================================================
TOP_K_RESULTS: int = 10
SIMILARITY_THRESHOLD: float = 0.5
EMBEDDING_SOURCE: str = "baseline"  # Options: "baseline", "transfer_learning", "siamese"
SIMILARITY_ALGORITHM: str = "cosine"  # Options: "cosine"
ENABLE_FAISS: bool = True

# ==============================================================================
# RESOURCE & ACCELERATOR CONFIGURATION
# ==============================================================================
DEVICE_CONFIGURATION: str = "GPU"  # fallback to CPU is handled automatically

# ==============================================================================
# LOGGING CONFIGURATIONS
# ==============================================================================
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
