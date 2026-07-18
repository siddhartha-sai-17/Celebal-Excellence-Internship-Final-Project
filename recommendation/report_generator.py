"""
Module Description: Recommendation Report Generator
Purpose: Saves system comparison results and benchmarks to JSON, CSV summary sheets, and generates visualization charts.
Author: Technical Lead
Version: 1.0.0
Dependencies: pandas, numpy, json, matplotlib, seaborn, config.settings, utils.logger
"""

import json
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import settings
from utils.logger import app_logger


class ReportGenerator:
    """
    Serializes comparison records, metrics, and profiles to filesystem reports and renders summary charts.
    """

    def __init__(self, output_dir: Path = settings.REPORTS_DIRECTORY) -> None:
        """
        Initializes the report generator.

        Args:
            output_dir: Folder to save JSON and CSV reports.
        """
        self.output_dir: Path = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir: Path = settings.PLOTS_DIRECTORY
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def save_comparison_report(self, comparison_data: Dict[str, Any]) -> None:
        """
        Saves the comparison dictionary to comparison_report.json and exports a CSV summary sheet.

        Args:
            comparison_data: Comparative dict mapping models to metrics.
        """
        app_logger.info("Saving comparison reports...")

        # 1. Save JSON comparison report
        json_path = self.output_dir / "comparison_report.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(comparison_data, f, indent=4)
            app_logger.info(f"Comparison report saved to {json_path}")
        except Exception as e:
            app_logger.error(f"Failed to save comparison JSON: {e}")

        # 2. Save individual reports
        for model, metrics in comparison_data.items():
            model_json_path = self.output_dir / f"{model}_report.json"
            try:
                with open(model_json_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=4)
            except Exception as e:
                app_logger.error(f"Failed to save {model} JSON report: {e}")

        # 3. Create CSV summary sheet
        csv_path = self.output_dir / "performance_summary.csv"
        try:
            # Flatten dict to tabular format
            rows = []
            for model, metrics in comparison_data.items():
                row = {"model_configuration": model}
                row.update(metrics)
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            app_logger.info(f"Performance summary sheet exported to {csv_path}")
        except Exception as e:
            app_logger.error(f"Failed to export summary CSV: {e}")

    def generate_comparison_plots(self, comparison_data: Dict[str, Any]) -> None:
        """
        Renders bar charts comparing Precision, Recall, and mAP across configurations.

        Args:
            comparison_data: Comparative dictionary mapping models to metrics.
        """
        app_logger.info("Generating comparative visualization plots...")

        try:
            rows = []
            for model, metrics in comparison_data.items():
                row = {"model": model}
                row.update(metrics)
                rows.append(row)
            df = pd.DataFrame(rows)

            # Define color palette
            sns.set_theme(style="whitegrid")

            # 1. Precision Comparison Bar Chart
            plt.figure(figsize=(10, 6))
            prec_cols = [c for c in df.columns if "precision_at_" in c]
            if prec_cols:
                df_melt = df.melt(id_vars=["model"], value_vars=prec_cols, var_name="Metric", value_name="Score")
                sns.barplot(data=df_melt, x="Metric", y="Score", hue="model", palette="muted")
                plt.title("Precision Comparison (Baseline vs Fine-Tuned vs Siamese)", fontsize=12, fontweight="bold")
                plt.ylim(0, 1.0)
                plt.tight_layout()
                plt.savefig(self.plots_dir / "precision_comparison.png", dpi=150)
                plt.close()

            # 2. Recall Comparison Bar Chart
            plt.figure(figsize=(10, 6))
            rec_cols = [c for c in df.columns if "recall_at_" in c]
            if rec_cols:
                df_melt = df.melt(id_vars=["model"], value_vars=rec_cols, var_name="Metric", value_name="Score")
                sns.barplot(data=df_melt, x="Metric", y="Score", hue="model", palette="muted")
                plt.title("Recall Comparison (Baseline vs Fine-Tuned vs Siamese)", fontsize=12, fontweight="bold")
                plt.ylim(0, 1.0)
                plt.tight_layout()
                plt.savefig(self.plots_dir / "recall_comparison.png", dpi=150)
                plt.close()

            # 3. mAP Comparison Bar Chart
            plt.figure(figsize=(8, 5))
            map_cols = [c for c in df.columns if "ap_at_" in c]
            if map_cols:
                df_melt = df.melt(id_vars=["model"], value_vars=map_cols, var_name="Metric", value_name="Score")
                # Clean metric labels
                df_melt["Metric"] = df_melt["Metric"].apply(lambda x: x.replace("ap_", "mAP_"))
                sns.barplot(data=df_melt, x="Metric", y="Score", hue="model", palette="muted")
                plt.title("Mean Average Precision (mAP) Comparison", fontsize=12, fontweight="bold")
                plt.ylim(0, 1.0)
                plt.tight_layout()
                plt.savefig(self.plots_dir / "map_comparison.png", dpi=150)
                plt.close()

            # 4. Latency Comparison Chart
            plt.figure(figsize=(8, 5))
            latency_col = "average_recommendation_time_sec"
            if latency_col in df.columns:
                sns.barplot(data=df, x="model", y=latency_col, palette="Set2")
                plt.title("Average Recommendation Response Latency", fontsize=12, fontweight="bold")
                plt.ylabel("Time (seconds)")
                plt.tight_layout()
                plt.savefig(self.plots_dir / "latency_comparison.png", dpi=150)
                plt.close()

            app_logger.info("Comparative visualization plots generated successfully.")
        except Exception as e:
            app_logger.error(f"Failed to generate comparative visualization plots: {e}")
