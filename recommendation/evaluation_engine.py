"""
Module Description: Information Retrieval Metrics Evaluator
Purpose: Computes Precision@K, Recall@K, mAP, MRR, Hit Rate, and NDCG to assess recommendation search quality.
Author: Technical Lead
Version: 1.0.0
Dependencies: numpy, pandas, config.settings, utils.logger
"""

import math
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from utils.logger import app_logger


class EvaluationEngine:
    """
    Computes information retrieval benchmarks to evaluate and compare embedding representations.
    """

    @staticmethod
    def compute_precision_at_k(retrieved_categories: List[str], query_category: str, k: int) -> float:
        """
        Calculates Precision@K = (Relevant Retrieved Items in Top K) / K.
        """
        if k <= 0:
            return 0.0
        top_k = retrieved_categories[:k]
        relevant_count = sum(1 for cat in top_k if cat == query_category)
        return relevant_count / k

    @staticmethod
    def compute_recall_at_k(retrieved_categories: List[str], 
                            query_category: str, 
                            k: int, 
                            total_relevant_in_db: int) -> float:
        """
        Calculates Recall@K = (Relevant Retrieved Items in Top K) / (Total Relevant Items in DB).
        """
        if total_relevant_in_db <= 0:
            return 1.0
        top_k = retrieved_categories[:k]
        relevant_count = sum(1 for cat in top_k if cat == query_category)
        return relevant_count / total_relevant_in_db

    @staticmethod
    def compute_average_precision(retrieved_categories: List[str], query_category: str, k: int) -> float:
        """
        Calculates Average Precision (AP) for a single query.
        """
        top_k = retrieved_categories[:k]
        relevant_indices = [i for i, cat in enumerate(top_k) if cat == query_category]
        
        if not relevant_indices:
            return 0.0

        ap_sum = 0.0
        for idx in relevant_indices:
            # Rank is 1-indexed
            rank = idx + 1
            precision_at_rank = sum(1 for i in range(rank) if top_k[i] == query_category) / rank
            ap_sum += precision_at_rank

        return ap_sum / len(relevant_indices)

    @staticmethod
    def compute_reciprocal_rank(retrieved_categories: List[str], query_category: str, k: int) -> float:
        """
        Calculates Reciprocal Rank (RR) = 1 / (Rank of first relevant item).
        """
        top_k = retrieved_categories[:k]
        for idx, cat in enumerate(top_k):
            if cat == query_category:
                return 1.0 / (idx + 1)
        return 0.0

    @staticmethod
    def compute_hit_rate(retrieved_categories: List[str], query_category: str, k: int) -> float:
        """
        Calculates Hit Rate = 1 if at least one relevant item is in Top K, else 0.
        """
        top_k = retrieved_categories[:k]
        return 1.0 if any(cat == query_category for cat in top_k) else 0.0

    @staticmethod
    def compute_ndcg_at_k(retrieved_categories: List[str], query_category: str, k: int) -> float:
        """
        Calculates Normalized Discounted Cumulative Gain (NDCG)@K using binary relevance.
        """
        top_k = retrieved_categories[:k]
        
        # Calculate DCG
        dcg = 0.0
        for idx, cat in enumerate(top_k):
            relevance = 1.0 if cat == query_category else 0.0
            dcg += relevance / math.log2(idx + 2)

        # Calculate Ideal DCG (all top K items are relevant)
        idcg = 0.0
        relevant_count = sum(1 for cat in top_k if cat == query_category)
        for idx in range(relevant_count):
            idcg += 1.0 / math.log2(idx + 2)

        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    @classmethod
    def evaluate_queries(cls, 
                         queries_meta: pd.DataFrame, 
                         recommendations_list: List[List[Dict[str, Any]]], 
                         db_meta: pd.DataFrame,
                         k_vals: List[int] = [5, 10, 20]) -> Dict[str, Any]:
        """
        Aggregates retrieval metrics over a set of queries.

        Args:
            queries_meta: DataFrame of query images details.
            recommendations_list: List of recommendations results matching queries.
            db_meta: Complete database metadata (for total relevant items counts).
            k_vals: Values of K to compute metrics for.

        Returns:
            Dictionary of averaged retrieval metrics (Precision@K, Recall@K, mAP@K, MRR, HitRate, NDCG).
        """
        app_logger.info(f"Aggregating evaluation metrics over {len(queries_meta)} queries...")
        
        metrics_accumulators: Dict[str, List[float]] = {}
        for k in k_vals:
            metrics_accumulators[f"precision_at_{k}"] = []
            metrics_accumulators[f"recall_at_{k}"] = []
            metrics_accumulators[f"ap_at_{k}"] = []
            metrics_accumulators[f"mrr_at_{k}"] = []
            metrics_accumulators[f"hit_rate_at_{k}"] = []
            metrics_accumulators[f"ndcg_at_{k}"] = []

        # Count total relevant items per category in database to calculate true recall
        category_db_counts = db_meta["articleType"].value_counts().to_dict()

        for idx, row in queries_meta.iterrows():
            query_cat = row["articleType"]
            recs = recommendations_list[idx]
            retrieved_cats = [r["category"] for r in recs]

            # Total relevant items in DB (subtracting 1 because query itself is in DB or excluded)
            total_relevant = max(1, category_db_counts.get(query_cat, 1) - 1)

            for k in k_vals:
                metrics_accumulators[f"precision_at_{k}"].append(
                    cls.compute_precision_at_k(retrieved_cats, query_cat, k)
                )
                metrics_accumulators[f"recall_at_{k}"].append(
                    cls.compute_recall_at_k(retrieved_cats, query_cat, k, total_relevant)
                )
                metrics_accumulators[f"ap_at_{k}"].append(
                    cls.compute_average_precision(retrieved_cats, query_cat, k)
                )
                metrics_accumulators[f"mrr_at_{k}"].append(
                    cls.compute_reciprocal_rank(retrieved_cats, query_cat, k)
                )
                metrics_accumulators[f"hit_rate_at_{k}"].append(
                    cls.compute_hit_rate(retrieved_cats, query_cat, k)
                )
                metrics_accumulators[f"ndcg_at_{k}"].append(
                    cls.compute_ndcg_at_k(retrieved_cats, query_cat, k)
                )

        # Average results
        final_summary: Dict[str, float] = {}
        for key, vals in metrics_accumulators.items():
            final_summary[key] = float(np.mean(vals)) if vals else 0.0

        app_logger.info(f"Query evaluation metrics computed successfully.")
        return final_summary
