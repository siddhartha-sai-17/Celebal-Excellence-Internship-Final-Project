"""
Module Description: Recommendation Ranking and Filtering Engine
Purpose: Ranks similarities, filters based on thresholds and attributes, deduplicates, and formats results.
Author: Technical Lead
Version: 1.0.0
Dependencies: pandas, numpy, config.settings, utils.logger
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from config import settings
from utils.logger import app_logger


class RankingEngine:
    """
    Applies constraints, sorts raw scores, filters recommendations, and formats response lists.
    """

    @staticmethod
    def rank_and_filter(scores: np.ndarray,
                        metadata_df: pd.DataFrame,
                        query_id: Optional[str] = None,
                        gender_filter: Optional[str] = None,
                        category_filter: Optional[str] = None,
                        color_filter: Optional[str] = None,
                        season_filter: Optional[str] = None,
                        usage_filter: Optional[str] = None,
                        similarity_threshold: float = settings.SIMILARITY_THRESHOLD,
                        top_k: int = settings.TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """
        Ranks similarity scores and applies user-defined metadata filters.

        Args:
            scores: Pairwise similarity score array of shape (num_samples,).
            metadata_df: Metadata matching the database index rows.
            query_id: Optional ID of the query image to exclude from results.
            gender_filter: Optional gender filter.
            category_filter: Optional category filter (articleType).
            color_filter: Optional color filter.
            season_filter: Optional season filter.
            usage_filter: Optional usage filter.
            similarity_threshold: Minimum similarity score required.
            top_k: Number of ranked matches to return.

        Returns:
            A list of dictionary cards formatted for presentation.
        """
        app_logger.info("Ranking and filtering recommendations...")

        # Construct raw candidates list
        candidates = []
        for idx, row in metadata_df.iterrows():
            score = float(scores[idx])
            # Filter by threshold
            if score < similarity_threshold:
                continue

            # Exclude query image if same ID
            product_id = str(row["id"])
            if query_id and product_id == query_id:
                continue

            candidates.append((score, row))

        # Sort by similarity score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        results: List[Dict[str, Any]] = []
        seen_ids = set()

        for score, row in candidates:
            # Deduplicate by ID
            product_id = str(row["id"])
            if product_id in seen_ids:
                continue

            # Apply metadata filters (case-insensitive)
            if gender_filter and str(row.get("gender", "")).lower() != gender_filter.lower():
                continue
            if category_filter and str(row.get("articleType", "")).lower() != category_filter.lower():
                continue
            if color_filter and str(row.get("baseColour", "")).lower() != color_filter.lower():
                continue
            if season_filter and str(row.get("season", "")).lower() != season_filter.lower():
                continue
            if usage_filter and str(row.get("usage", "")).lower() != usage_filter.lower():
                continue

            seen_ids.add(product_id)

            # Format result card
            percentage = round(score * 100, 2)
            card = {
                "rank": len(results) + 1,
                "id": product_id,
                "similarity_score": score,
                "similarity_percentage": f"{percentage}%",
                "product_name": row.get("displayName", "Unknown Product"),
                "category": row.get("articleType", "Unknown"),
                "subcategory": row.get("subCategory", "Unknown"),
                "gender": row.get("gender", "Unknown"),
                "color": row.get("baseColour", "Unknown"),
                "season": row.get("season", "Unknown"),
                "usage": row.get("usage", "Unknown"),
                "image_path": row.get("image_path", "")
            }
            results.append(card)

            # Stop once top_k is reached
            if len(results) >= top_k:
                break

        app_logger.info(f"Retrieved {len(results)} recommendations matching filters.")
        return results
