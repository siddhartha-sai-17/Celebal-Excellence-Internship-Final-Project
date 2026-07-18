"""
Module Description: Siamese Pair Generation Utilities
Purpose: Dynamically constructs balanced positive and negative image pairs for metric learning.
Author: Technical Lead
Version: 1.0.0
Dependencies: pandas, numpy, random, config.settings, utils.logger
"""

import random
from typing import List, Tuple, Set, Dict
import numpy as np
import pandas as pd
from config import settings
from utils.logger import app_logger


def generate_balanced_pairs(df: pd.DataFrame) -> Tuple[List[Tuple[str, str]], List[int]]:
    """
    Generates balanced positive (same class) and negative (different class) image pairs.

    Args:
        df: DataFrame containing the dataset subset with columns 'id' and 'articleType'.

    Returns:
        A tuple of (pairs, labels) where:
        - pairs is a list of tuples containing (image_path_1, image_path_2).
        - labels is a list of integers (1 for positive pair, 0 for negative pair).
    """
    app_logger.info("Generating balanced image pairs for Siamese training...")
    random.seed(settings.RANDOM_SEED)

    # Group image paths by category (articleType)
    category_groups: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        cat = row["articleType"]
        img_path = row["image_path"]
        if cat not in category_groups:
            category_groups[cat] = []
        category_groups[cat].append(img_path)

    categories = list(category_groups.keys())
    if len(categories) < 2:
        msg = f"At least 2 distinct categories are required to generate negative pairs. Found categories: {categories}"
        app_logger.error(msg)
        raise ValueError(msg)

    # Determine target number of pairs based on dataset size
    # Let's generate about 5 times the dataset size to have rich pairs, e.g. ~10,000 pairs
    num_samples = len(df)
    target_pairs_per_class = min(1000, num_samples)

    positive_pairs: Set[Tuple[str, str]] = set()
    negative_pairs: Set[Tuple[str, str]] = set()

    # Generate Positive Pairs (same category)
    app_logger.info("Generating positive pairs...")
    for cat, paths in category_groups.items():
        if len(paths) < 2:
            continue
        # Generate pairs for this category
        attempts = 0
        max_attempts = len(paths) * 10
        cat_pos_count = 0
        target = target_pairs_per_class // len(categories)

        while cat_pos_count < target and attempts < max_attempts:
            attempts += 1
            img1, img2 = random.sample(paths, 2)
            # Order to prevent duplicates (img1, img2) vs (img2, img1)
            pair = (img1, img2) if img1 < img2 else (img2, img1)
            if pair not in positive_pairs:
                positive_pairs.add(pair)
                cat_pos_count += 1

    # Generate Negative Pairs (different categories)
    app_logger.info("Generating negative pairs...")
    target_negatives = len(positive_pairs)
    attempts = 0
    max_attempts = target_negatives * 20
    neg_count = 0

    while neg_count < target_negatives and attempts < max_attempts:
        attempts += 1
        cat1, cat2 = random.sample(categories, 2)
        img1 = random.choice(category_groups[cat1])
        img2 = random.choice(category_groups[cat2])
        pair = (img1, img2) if img1 < img2 else (img2, img1)
        if pair not in negative_pairs:
            negative_pairs.add(pair)
            neg_count += 1

    # Balance positive and negative counts
    final_pos_list = list(positive_pairs)
    final_neg_list = list(negative_pairs)

    min_size = min(len(final_pos_list), len(final_neg_list))
    final_pos_list = final_pos_list[:min_size]
    final_neg_list = final_neg_list[:min_size]

    # Combine pairs and labels
    pairs = final_pos_list + final_neg_list
    # 1 for positive, 0 for negative
    labels = [1] * len(final_pos_list) + [0] * len(final_neg_list)

    # Shuffle combined dataset using random seed
    combined = list(zip(pairs, labels))
    random.shuffle(combined)
    
    shuffled_pairs = [item[0] for item in combined]
    shuffled_labels = [item[1] for item in combined]

    app_logger.info(f"Generated {len(shuffled_pairs)} pairs: {min_size} positive, {min_size} negative.")
    return shuffled_pairs, shuffled_labels
