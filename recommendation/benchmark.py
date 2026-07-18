"""
Module Description: Recommendation Benchmarking
Purpose: Measures embedding generation times, search latency, cold start vs warm start speeds, and RAM consumption.
Author: Technical Lead
Version: 1.0.0
Dependencies: psutil, time, numpy, config.settings, utils.logger, recommendation.recommendation_engine
"""

import time
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from config import settings
from utils.logger import performance_logger, exception_logger
from recommendation.recommendation_engine import RecommendationEngine

# Optional psutil import for memory profiling
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    performance_logger.warning("psutil library not found. System resource usage metrics will be estimated.")


class RecommendationBenchmark:
    """
    Profiles recommendation latency, memory footprint, and frames-per-second (FPS) throughput.
    """

    def __init__(self) -> None:
        """
        Initializes the benchmark loader.
        """
        self.engine = RecommendationEngine()

    @staticmethod
    def get_memory_usage_mb() -> float:
        """
        Returns the current memory usage of the process in Megabytes.
        """
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        return 0.0

    def run_latency_benchmark(self, 
                              sample_images: List[str], 
                              model_type: str, 
                              runs: int = 10) -> Dict[str, Any]:
        """
        Profiles warm and cold start retrieval speeds, computing latency distributions.

        Args:
            sample_images: List of image paths to use for testing.
            model_type: Target embedding model ("baseline", "transfer_learning", "siamese").
            runs: Number of queries to process.

        Returns:
            Dictionary containing latency stats.
        """
        performance_logger.info(f"Starting latency benchmarking for model type '{model_type}' over {runs} runs...")
        
        if not sample_images:
            performance_logger.error("No sample images provided for benchmarking.")
            return {}

        latencies_inf = []
        latencies_sim = []
        latencies_total = []
        memory_usages = []

        # 1. Cold Start Run (forces model loading and first inference setup)
        memory_before = self.get_memory_usage_mb()
        t_cold_start = time.perf_counter()
        
        # Select first sample image
        test_img = sample_images[0]
        _, _ = self.engine.recommend(test_img, model_type=model_type)
        
        cold_start_latency = time.perf_counter() - t_cold_start
        memory_after = self.get_memory_usage_mb()
        memory_leak = memory_after - memory_before

        performance_logger.info(f"Cold Start Latency for {model_type}: {cold_start_latency:.4f}s. Memory delta: {memory_leak:.2f}MB")

        # 2. Warm Runs (measuring steady state performance)
        for i in range(runs):
            # Select random or sequential image from test set
            img = sample_images[i % len(sample_images)]
            
            memory_before = self.get_memory_usage_mb()
            
            t_run = time.perf_counter()
            _, lat_metrics = self.engine.recommend(img, model_type=model_type)
            total_time = time.perf_counter() - t_run
            
            latencies_inf.append(lat_metrics.get("inference", 0.0))
            latencies_sim.append(lat_metrics.get("similarity_search", 0.0))
            latencies_total.append(total_time)
            
            memory_usages.append(self.get_memory_usage_mb())

        # Compute stats
        fps = 1.0 / np.mean(latencies_total) if latencies_total else 0.0
        
        summary = {
            "model_type": model_type,
            "cold_start_latency_sec": float(cold_start_latency),
            "average_inference_latency_sec": float(np.mean(latencies_inf)),
            "average_similarity_search_latency_sec": float(np.mean(latencies_sim)),
            "average_recommendation_latency_sec": float(np.mean(latencies_total)),
            "p50_latency_sec": float(np.percentile(latencies_total, 50)),
            "p95_latency_sec": float(np.percentile(latencies_total, 95)),
            "average_memory_usage_mb": float(np.mean(memory_usages)),
            "peak_memory_usage_mb": float(np.max(memory_usages)) if memory_usages else 0.0,
            "inference_fps": float(fps)
        }

        performance_logger.info(f"Benchmark summary for {model_type}: {summary}")
        return summary
