"""
Module Description: Recommendation Ranking and Filtering Engine
Purpose: Ranks similarities, filters based on thresholds and attributes, deduplicates, and formats results.
Author: Technical Lead
Version: 2.0.0
Dependencies: pandas, numpy, config.settings, utils.logger

Key improvements in v2.0:
- Category/metadata filters are applied FIRST before threshold pruning.
  This ensures that when the user explicitly filters by category (e.g. "Watches"),
  the top-K results are always from that category, regardless of cross-category
  embedding proximity.
- Auto-fallback: if fewer than top_k results pass the threshold, the threshold is
  progressively relaxed (to a minimum of 0.10) until enough results are returned.
- Prevents the common failure mode where a query image from a minority category
  (e.g. watches) gets zero results because the global threshold prunes all category
  matches that scored lower than cross-category matches.
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

        IMPORTANT — filter order:
          1. Exclude query image itself.
          2. Apply all metadata filters (category, color, gender, season, usage).
          3. Apply similarity threshold.
          4. Auto-relax threshold if not enough results are found.
          5. Sort by score and truncate to top_k.

        This ordering means explicit category filters are always honoured, even when
        the threshold would otherwise exclude all matching items.

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

        # ── Step 1: Attach scores to metadata rows ──────────────────────────
        scored = []
        for idx, row in metadata_df.iterrows():
            product_id = str(row["id"])
            # Exclude query image
            if query_id and product_id == query_id:
                continue

            score = float(scores[idx])

            # ── Step 2: Apply metadata filters FIRST ────────────────────────
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

            scored.append((score, product_id, row))

        # Sort all category-filtered candidates by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # ── Step 3: Apply threshold with auto-fallback ───────────────────────
        # When a category filter is active, relax threshold progressively so the
        # user always gets results for the category they asked for.
        effective_threshold = similarity_threshold
        min_threshold = 0.10  # hard floor — never go below 10% similarity

        while effective_threshold >= min_threshold:
            filtered = [(s, pid, row) for s, pid, row in scored if s >= effective_threshold]
            if len(filtered) >= min(3, top_k):  # need at least 3 results (or top_k if smaller)
                break
            if effective_threshold <= min_threshold:
                break
            # Relax by 0.10 each step
            effective_threshold = max(min_threshold, effective_threshold - 0.10)
            app_logger.info(
                f"Auto-relaxing similarity threshold to {effective_threshold:.2f} "
                f"(only {len(filtered)} results at previous threshold)"
            )

        if effective_threshold < similarity_threshold:
            app_logger.info(
                f"Threshold relaxed from {similarity_threshold:.2f} → {effective_threshold:.2f} "
                f"to surface category-filtered results."
            )

        # ── Step 4: Build final results list ────────────────────────────────
        results: List[Dict[str, Any]] = []
        seen_ids: set = set()

        for score, product_id, row in filtered:
            if product_id in seen_ids:
                continue
            seen_ids.add(product_id)

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

            if len(results) >= top_k:
                break

        app_logger.info(f"Retrieved {len(results)} recommendations matching filters.")
        return results
