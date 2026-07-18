"""
Module Description: Siamese Triplet Generation Utilities
Purpose: Constructs image triplets (Anchor, Positive, Negative) for Metric Learning.
Author: Technical Lead
Version: 1.0.0
Dependencies: pandas, numpy, random, config.settings, utils.logger
"""

import random
from typing import List, Tuple, Dict
import pandas as pd
from config import settings
from utils.logger import app_logger


def generate_triplets(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """
    Generates triplets (Anchor, Positive, Negative) for training.

    Args:
        df: DataFrame containing the dataset subset with columns 'image_path' and 'articleType'.

    Returns:
        A list of tuples: (anchor_path, positive_path, negative_path).
    """
    app_logger.info("Generating triplets for Siamese network training...")
    random.seed(settings.RANDOM_SEED)

    # Group image paths by category
    category_groups: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        cat = row["articleType"]
        img_path = row["image_path"]
        if cat not in category_groups:
            category_groups[cat] = []
        category_groups[cat].append(img_path)

    categories = list(category_groups.keys())
    if len(categories) < 2:
        msg = f"At least 2 categories required for triplet generation. Found: {categories}"
        app_logger.error(msg)
        raise ValueError(msg)

    triplets: List[Tuple[str, str, str]] = []
    
    # Target size: 2-3 triplets per image in the dataset to expand training rich samples
    num_samples = len(df)
    target_triplets_count = num_samples * 2

    # Loop over all anchor candidates
    attempts = 0
    max_attempts = target_triplets_count * 10
    
    while len(triplets) < target_triplets_count and attempts < max_attempts:
        attempts += 1
        # Select anchor category and image
        anchor_cat = random.choice(categories)
        if len(category_groups[anchor_cat]) < 2:
            continue
        
        anchor_path, positive_path = random.sample(category_groups[anchor_cat], 2)
        
        # Select negative category and image
        negative_cat = random.choice([c for c in categories if c != anchor_cat])
        negative_path = random.choice(category_groups[negative_cat])
        
        triplets.append((anchor_path, positive_path, negative_path))

    # Shuffle triplets
    random.shuffle(triplets)
    
    app_logger.info(f"Generated {len(triplets)} triplets successfully.")
    return triplets
