"""
Module Description: Unified Recommendation Engine
Purpose: Provides the unified interface to retrieve recommendations using any of the three models, with filtering and FAISS.
Author: Technical Lead
Version: 1.0.0
Dependencies: recommendation.recommendation_service, config.settings, utils.logger
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from PIL import Image
from config import settings
from utils.logger import recommendation_logger
from recommendation.recommendation_service import RecommendationService


class RecommendationEngine:
    """
    A unified visual search engine wrapper consuming modular recommendation service components.
    """

    def __init__(self) -> None:
        """
        Initializes the recommendation engine.
        """
        self.service = RecommendationService()

    def recommend(self,
                  image_input: Union[str, Path, Image.Image],
                  model_type: Optional[str] = None,
                  gender_filter: Optional[str] = None,
                  category_filter: Optional[str] = None,
                  color_filter: Optional[str] = None,
                  season_filter: Optional[str] = None,
                  usage_filter: Optional[str] = None,
                  similarity_threshold: Optional[float] = None,
                  top_k: Optional[int] = None,
                  use_faiss: Optional[bool] = None) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Main Visual Recommendation Entry Point. Retrieves visually similar product cards.

        Args:
            image_input: Query image path or PIL Image object.
            model_type: Selected model ("baseline", "transfer_learning", "siamese"). Uses config if None.
            gender_filter: Filtering by gender category.
            category_filter: Filtering by articleType.
            color_filter: Filtering by baseColour.
            season_filter: Filtering by season.
            usage_filter: Filtering by usage.
            similarity_threshold: Similarity score limit.
            top_k: Number of recommendations to retrieve.
            use_faiss: Toggle FAISS search acceleration.

        Returns:
            A tuple of (recommendations_list, latency_metrics_dict).
        """
        # Apply defaults from settings if parameters are omitted
        selected_model = model_type or settings.EMBEDDING_SOURCE
        thresh = similarity_threshold if similarity_threshold is not None else settings.SIMILARITY_THRESHOLD
        k = top_k if top_k is not None else settings.TOP_K_RESULTS
        faiss_flag = use_faiss if use_faiss is not None else settings.ENABLE_FAISS

        recommendation_logger.info(
            f"Unified visual recommendation requested. Model: {selected_model}, Threshold: {thresh}, Top-K: {k}, FAISS: {faiss_flag}"
        )

        # Delegate query to recommendation service
        return self.service.get_recommendations(
            image_input=image_input,
            source_name=selected_model,
            gender_filter=gender_filter,
            category_filter=category_filter,
            color_filter=color_filter,
            season_filter=season_filter,
            usage_filter=usage_filter,
            similarity_threshold=thresh,
            top_k=k,
            use_faiss=faiss_flag
        )
