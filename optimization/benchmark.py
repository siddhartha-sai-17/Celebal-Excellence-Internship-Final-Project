"""
Module Description: Optimization Benchmarking
Purpose: Profiles cached queries versus direct inference passes.
Author: Technical Lead
Version: 1.0.0
Dependencies: time, config.settings, utils.logger, recommendation.benchmark, optimization.recommendation_optimizer
"""

import time
from typing import Dict, Any, List
from config import settings
from utils.logger import performance_logger
from recommendation.benchmark import RecommendationBenchmark
from optimization.recommendation_optimizer import RecommendationOptimizer


class OptimizationBenchmark:
    """
    Measures performance boosts from indexing and query caching.
    """

    @staticmethod
    def compare_caching_gain(sample_img: str, runs: int = 100) -> Dict[str, float]:
        """
        Compares query latency of cached hits versus uncached evaluations.

        Args:
            sample_img: Target test image path.
            runs: Number of queries.

        Returns:
            Dictionary showing times and speedup multipliers.
        """
        from recommendation.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        
        # Cold start (first time)
        t0 = time.perf_counter()
        res = engine.recommend(sample_img, model_type=settings.EMBEDDING_SOURCE)
        cold_time = time.perf_counter() - t0

        # Cache it
        RecommendationOptimizer.cache_recommendation(sample_img, settings.EMBEDDING_SOURCE, settings.TOP_K_RESULTS, res)

        # Measure cached runs
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = RecommendationOptimizer.get_cached_recommendation(sample_img, settings.EMBEDDING_SOURCE, settings.TOP_K_RESULTS)
        cached_total_time = time.perf_counter() - t0
        avg_cached_time = cached_total_time / runs

        speedup = cold_time / avg_cached_time if avg_cached_time > 0 else 1.0
        
        metrics = {
            "uncached_query_sec": float(cold_time),
            "avg_cached_query_sec": float(avg_cached_time),
            "speedup_multiplier": float(speedup)
        }

        performance_logger.info(f"[Optimization Benchmark] Caching comparison results: {metrics}")
        return metrics
