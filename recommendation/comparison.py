"""
Module Description: System Comparison Module
Purpose: Compares Baseline, Transfer Learning, and Siamese model embeddings side-by-side on identical queries.
Author: Technical Lead
Version: 1.0.0
Dependencies: pandas, numpy, config.settings, utils.logger, recommendation.recommendation_engine, recommendation.evaluation_engine
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from config import settings
from utils.logger import app_logger
from recommendation.recommendation_engine import RecommendationEngine
from recommendation.evaluation_engine import EvaluationEngine


class SystemComparison:
    """
    Performs systematic comparative auditing across baseline, fine-tuned, and metric-learned configurations.
    """

    def __init__(self, subset_metadata_path: Path = settings.SUBSET_DIRECTORY / "subset_metadata.csv") -> None:
        """
        Initializes comparison evaluator.

        Args:
            subset_metadata_path: Path to subset_metadata.csv.
        """
        self.subset_metadata_path: Path = subset_metadata_path
        self.engine = RecommendationEngine()

    def run_comparative_evaluation(self, num_queries: int = 50) -> Dict[str, Any]:
        """
        Runs top-K query searches for all three embedding models and computes comparative IR metrics.

        Args:
            num_queries: Number of query images to randomly select for testing.

        Returns:
            Dictionary comparing metrics of baseline, transfer_learning, and siamese.
        """
        app_logger.info(f"Running comparative evaluation on {num_queries} queries...")
        
        if not self.subset_metadata_path.exists():
            msg = f"Subset metadata file not found at {self.subset_metadata_path}."
            app_logger.error(msg)
            raise FileNotFoundError(msg)

        # Load subset metadata
        df = pd.read_csv(self.subset_metadata_path, on_bad_lines='skip')
        
        # Check files exist
        valid_rows = []
        for _, row in df.iterrows():
            if Path(row["image_path"]).exists():
                valid_rows.append(row)
        df_valid = pd.DataFrame(valid_rows)

        if len(df_valid) == 0:
            app_logger.error("No valid images found for comparative evaluation.")
            return {}

        # Sample test queries (stratified sampling if possible, else random)
        sample_size = min(num_queries, len(df_valid))
        queries_df = df_valid.sample(n=sample_size, random_state=settings.RANDOM_SEED).reset_index(drop=True)
        app_logger.info(f"Selected {len(queries_df)} query images for comparative analysis.")

        models_to_test = ["baseline", "transfer_learning", "siamese"]
        comparison_results: Dict[str, Any] = {}

        for model in models_to_test:
            app_logger.info(f"Evaluating model configuration '{model}'...")
            recs_list = []
            latencies = []

            for _, row in queries_df.iterrows():
                img_path = row["image_path"]
                # Query recommendation engine
                t0 = time.perf_counter()
                recs, _ = self.engine.recommend(
                    image_input=img_path,
                    model_type=model,
                    top_k=20,  # Retrieve top 20 to compute Precision@5, 10, 20
                    use_faiss=False  # Disable FAISS during accuracy audits to prevent minor index discrepancies
                )
                latencies.append(time.perf_counter() - t0)
                recs_list.append(recs)

            # Compute metrics
            summary_metrics = EvaluationEngine.evaluate_queries(
                queries_meta=queries_df,
                recommendations_list=recs_list,
                db_meta=df_valid,
                k_vals=[5, 10, 20]
            )

            # Add average latency
            summary_metrics["average_recommendation_time_sec"] = float(np.mean(latencies))
            
            comparison_results[model] = summary_metrics

        app_logger.info("Comparative evaluation completed successfully.")
        return comparison_results
