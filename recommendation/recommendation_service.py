"""
Module Description: Recommendation Coordinator Service
Purpose: Orchestrates the visual recommendation pipeline: loads query, runs CNN inference, matches index, ranks, and returns.
Author: Technical Lead
Version: 1.0.0
Dependencies: tensorflow, numpy, pandas, PIL.Image, config.settings, utils.logger, utils.timer
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from PIL import Image
import numpy as np
import tensorflow as tf
from config import settings
from utils.logger import recommendation_logger, exception_logger
from utils.timer import Timer
from preprocessing.image_preprocessor import ImagePreprocessor
from recommendation.embedding_selector import EmbeddingSelector
from recommendation.cache_manager import CacheManager
from recommendation.similarity_engine import SimilarityEngine
from recommendation.ranking_engine import RankingEngine


class RecommendationService:
    """
    Provides visual search recommendations based on configured embedding sources.
    """

    def __init__(self) -> None:
        """
        Initializes the service and fetches the CacheManager instance.
        """
        self.cache: CacheManager = CacheManager()

    def get_recommendations(self, 
                            image_input: Union[str, Path, Image.Image],
                            source_name: str = settings.EMBEDDING_SOURCE,
                            gender_filter: Optional[str] = None,
                            category_filter: Optional[str] = None,
                            color_filter: Optional[str] = None,
                            season_filter: Optional[str] = None,
                            usage_filter: Optional[str] = None,
                            similarity_threshold: float = settings.SIMILARITY_THRESHOLD,
                            top_k: int = settings.TOP_K_RESULTS,
                            use_faiss: bool = settings.ENABLE_FAISS) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Processes an uploaded image query and returns ranked visual recommendations.

        Args:
            image_input: PIL image or file path of the query image.
            source_name: Selection of model embeddings ("baseline", "transfer_learning", "siamese").
            gender_filter: Metadata filter for gender.
            category_filter: Metadata filter for product category.
            color_filter: Metadata filter for product color.
            season_filter: Metadata filter for season.
            usage_filter: Metadata filter for usage.
            similarity_threshold: Minimum similarity value.
            top_k: Number of recommendations to retrieve.
            use_faiss: If True, uses FAISS index for matching (if available).

        Returns:
            A tuple of (recommendations_list, latency_metrics_dict).
        """
        recommendation_logger.info(
            f"Recommendation request started using source: {source_name}. "
            f"Filters: gender={gender_filter}, category={category_filter}, "
            f"color={color_filter}, season={season_filter}, usage={usage_filter}, "
            f"threshold={similarity_threshold}, FAISS={use_faiss}"
        )
        start_time = time.perf_counter()
        
        latencies: Dict[str, float] = {
            "image_loading": 0.0,
            "preprocessing": 0.0,
            "inference": 0.0,
            "similarity_search": 0.0,
            "ranking": 0.0,
            "total_elapsed": 0.0
        }

        # 1. Resolve embedding source files and checkpoints
        emb_dir, checkpoint_path = EmbeddingSelector.get_paths(source_name)

        # 2. Preprocess query image
        t_pre = time.perf_counter()
        img_tensor = ImagePreprocessor.preprocess_for_inference(image_input)
        latencies["preprocessing"] = time.perf_counter() - t_pre

        if img_tensor is None:
            msg = "Query image preprocessing failed. Unable to generate recommendations."
            recommendation_logger.error(msg)
            return [], latencies

        # 3. Load active model
        t_model_start = time.perf_counter()
        model = self.cache.get_model(source_name, checkpoint_path)
        if model is None:
            msg = f"Failed to retrieve or build the model for source {source_name}"
            recommendation_logger.error(msg)
            return [], latencies

        # 4. Extract Query Embedding (CNN Inference)
        t_inf = time.perf_counter()
        try:
            # Generate embedding using the model
            query_emb_tensor = model(img_tensor, training=False)
            query_emb = query_emb_tensor.numpy().squeeze()
            latencies["inference"] = time.perf_counter() - t_inf
        except Exception as e:
            exception_logger.error(f"Inference execution failed on model: {e}")
            return [], latencies

        # 5. Load Embedding Database
        db_embs, db_meta = self.cache.get_database(source_name, emb_dir)
        if db_embs is None or db_meta is None:
            msg = f"Failed to retrieve or load the embedding database for source {source_name}"
            recommendation_logger.error(msg)
            return [], latencies

        # 6. Similarity Search
        t_sim = time.perf_counter()
        scores = None

        # When metadata filters are active, FAISS is counterproductive: it returns
        # only the global top-K nearest neighbors, which may entirely exclude the
        # filtered category (e.g. all top-100 FAISS hits are shoes when user asks
        # for Watches → 0 results). With only 1,750 vectors, full cosine similarity
        # over the entire database takes < 1ms, so FAISS overhead is not needed.
        has_metadata_filter = any([
            gender_filter, category_filter, color_filter, season_filter, usage_filter
        ])

        if use_faiss and not has_metadata_filter:
            # FAISS is only beneficial for unfiltered searches over large collections
            faiss_eng = self.cache.get_faiss_engine(source_name, db_embs)
            if faiss_eng is not None:
                faiss_res = faiss_eng.search(query_emb, top_k=min(top_k * 10, len(db_embs)))
                if faiss_res is not None:
                    sims, idxs = faiss_res
                    scores = np.zeros(len(db_embs), dtype=np.float32)
                    for s, idx in zip(sims, idxs):
                        if idx >= 0:
                            scores[idx] = s

        if scores is None:
            # Full vectorised cosine similarity — used for all filtered searches
            # and as FAISS fallback for unfiltered searches
            scores = SimilarityEngine.compute_similarity(query_emb, db_embs, metric=settings.SIMILARITY_ALGORITHM)

        latencies["similarity_search"] = time.perf_counter() - t_sim


        # 7. Rank and Filter results
        t_rank = time.perf_counter()
        
        # Deduct query_id if image_input was a path (matching filename)
        query_id = None
        if isinstance(image_input, (str, Path)):
            query_id = Path(image_input).stem

        recommendations = RankingEngine.rank_and_filter(
            scores=scores,
            metadata_df=db_meta,
            query_id=query_id,
            gender_filter=gender_filter,
            category_filter=category_filter,
            color_filter=color_filter,
            season_filter=season_filter,
            usage_filter=usage_filter,
            similarity_threshold=similarity_threshold,
            top_k=top_k
        )
        latencies["ranking"] = time.perf_counter() - t_rank
        latencies["total_elapsed"] = time.perf_counter() - start_time

        recommendation_logger.info(f"Recommendation generation completed. Total time: {latencies['total_elapsed']:.4f} seconds.")
        return recommendations, latencies
