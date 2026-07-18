"""
Module Description: Recommendation Optimizer Layer
Purpose: Caches visual search outputs to avoid redundant CNN inference and similarity searches.
Author: Technical Lead
Version: 1.0.0
Dependencies: hashlib, typing, config.settings, utils.logger
"""

import hashlib
from typing import Dict, Any, List, Tuple, Union, Optional
from pathlib import Path
from PIL import Image
from utils.logger import performance_logger


class RecommendationOptimizer:
    """
    Maintains a local cache of recent visual recommendation results.
    """
    _results_cache: Dict[str, Tuple[List[Dict[str, Any]], Dict[str, float]]] = {}

    @classmethod
    def _get_query_hash(cls, 
                         image_input: Union[str, Path, Image.Image], 
                         model_type: str, 
                         top_k: int) -> str:
        """
        Generates a unique hash string based on query inputs.
        """
        if isinstance(image_input, (str, Path)):
            img_key = str(Path(image_input).resolve())
        elif isinstance(image_input, Image.Image):
            # Generate hash of image pixels
            img_bytes = image_input.tobytes()
            img_key = hashlib.md5(img_bytes).hexdigest()
        else:
            img_key = str(id(image_input))

        key = f"{img_key}_{model_type}_{top_k}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    @classmethod
    def get_cached_recommendation(cls, 
                                  image_input: Union[str, Path, Image.Image], 
                                  model_type: str, 
                                  top_k: int) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, float]]]:
        """
        Checks and returns cached recommendations if present.
        """
        qh = cls._get_query_hash(image_input, model_type, top_k)
        if qh in cls._results_cache:
            performance_logger.info(f"[Recommendation Cache Hit] Retrieved query match from RAM cache.")
            return cls._results_cache[qh]
        return None

    @classmethod
    def cache_recommendation(cls, 
                             image_input: Union[str, Path, Image.Image], 
                             model_type: str, 
                             top_k: int, 
                             results: Tuple[List[Dict[str, Any]], Dict[str, float]]) -> None:
        """
        Caches a recommendation result.
        """
        qh = cls._get_query_hash(image_input, model_type, top_k)
        cls._results_cache[qh] = results
        # Evict cache if too large (e.g. limit to 100 entries)
        if len(cls._results_cache) > 100:
            # Pop first item
            first_key = next(iter(cls._results_cache))
            cls._results_cache.pop(first_key)
siamese_optimizer = RecommendationOptimizer()
