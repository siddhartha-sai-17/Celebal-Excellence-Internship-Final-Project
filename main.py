"""
Module Description: Project CLI Entry Point
Purpose: Command-line interface to run dataset preparation, training, embedding generation, evaluation, and system benchmarking.
Author: Technical Lead
Version: 1.0.0
Dependencies: argparse, config.settings, utils.logger, preprocessing.dataset_subset, preprocessing.dataset_validator
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from config import settings
from utils.logger import app_logger, exception_logger
from preprocessing.dataset_subset import DatasetSubsetGenerator
from preprocessing.dataset_validator import DatasetValidator
from models.model_factory import ModelFactory
from models.checkpoint_manager import CheckpointManager
from models.model_loader import ModelLoader
from recommendation.embedding_database import EmbeddingDatabase
from preprocessing.image_preprocessor import ImagePreprocessor


def run_dataset_preparation() -> None:
    """
    Extracts the balanced category subset from the large zip file.
    """
    app_logger.info("Starting Dataset Preparation...")
    try:
        # Check zip archive existence
        zip_file = Path(r"C:\Users\banda\Downloads\archive (3).zip")
        if not zip_file.exists():
            app_logger.error(f"Raw dataset zip not found at {zip_file}")
            sys.exit(1)

        generator = DatasetSubsetGenerator(zip_path=zip_file)
        df_subset = generator.generate_subset()
        
        # Save statistics report
        stats = {
            "total_images": len(df_subset),
            "categories": settings.SELECTED_CATEGORIES,
            "images_per_category": df_subset["articleType"].value_counts().to_dict()
        }
        stats_path = settings.EVALUATION_DIRECTORY / "dataset_statistics.json"
        import json
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)
        app_logger.info(f"Dataset statistics exported to {stats_path}")

        # Plot distribution
        from utils.visualization import plot_category_distribution
        plot_category_distribution(df_subset)
        
        app_logger.info("Dataset Preparation completed successfully.")
    except Exception as e:
        exception_logger.error(f"Dataset preparation failed: {e}")
        sys.exit(1)


def run_dataset_validation() -> None:
    """
    Runs integrity validation check on the extracted subset.
    """
    app_logger.info("Starting Dataset Validation...")
    try:
        csv_path = settings.SUBSET_DIRECTORY / "subset_metadata.csv"
        images_dir = settings.SUBSET_DIRECTORY / "images"

        if not csv_path.exists() or not images_dir.exists():
            app_logger.error("Subset metadata or images folder not found. Please run dataset preparation first.")
            sys.exit(1)

        validator = DatasetValidator(metadata_path=csv_path, images_dir=images_dir)
        report_path = settings.EVALUATION_DIRECTORY / "dataset_validation_report.json"
        validator.save_report(report_path)
        app_logger.info("Dataset Validation completed successfully.")
    except Exception as e:
        exception_logger.error(f"Dataset validation failed: {e}")
        sys.exit(1)


def generate_embeddings(model_type: str) -> None:
    """
    Generates and saves visual embeddings for all images in the subset.

    Args:
        model_type: Selected model configuration ("baseline", "transfer_learning", "siamese").
    """
    app_logger.info(f"Starting embedding generation for model type: {model_type}...")
    try:
        from recommendation.embedding_selector import EmbeddingSelector
        emb_dir, checkpoint_path = EmbeddingSelector.get_paths(model_type)

        # Initialize base model
        model = ModelLoader.load_embedding_model(
            model_name=settings.MODEL_NAME,
            checkpoint_dir=checkpoint_path
        )
        if model is None:
            app_logger.error(f"Could not load or build the model for source: {model_type}")
            sys.exit(1)

        # Load subset metadata
        csv_path = settings.SUBSET_DIRECTORY / "subset_metadata.csv"
        if not csv_path.exists():
            app_logger.error("Subset metadata file not found. Run dataset preparation first.")
            sys.exit(1)

        df = pd.read_csv(csv_path, on_bad_lines='skip')
        
        # Filter existing image paths
        valid_rows = []
        for idx, row in df.iterrows():
            img_path = Path(row["image_path"])
            if not img_path.is_absolute():
                img_path = settings.BASE_DIR / img_path
            if img_path.exists():
                valid_rows.append(row)
        df_valid = pd.DataFrame(valid_rows)

        if len(df_valid) == 0:
            app_logger.error("No valid image files found on disk for embedding generation.")
            sys.exit(1)

        paths = df_valid["image_path"].tolist()

        app_logger.info(f"Extracting embeddings for {len(paths)} valid images using PIL batching...")
        
        embeddings_list = []
        batch_images = []
        
        for idx, path_str in enumerate(paths):
            try:
                from PIL import Image
                full_path = Path(path_str)
                if not full_path.is_absolute():
                    full_path = settings.BASE_DIR / full_path
                img = Image.open(full_path).convert("RGB")
                img = img.resize(settings.IMAGE_SIZE)
                img_arr = np.array(img, dtype=np.float32) / 255.0
                img_normalized = (img_arr - settings.IMAGENET_MEAN) / settings.IMAGENET_STD
                batch_images.append(img_normalized)
                
                # Batch prediction
                if len(batch_images) == 16 or idx == len(paths) - 1:
                    batch_tensor = tf.convert_to_tensor(batch_images, dtype=tf.float32)
                    batch_embs = model(batch_tensor, training=False).numpy()
                    embeddings_list.append(batch_embs)
                    batch_images = []
                    
                    if (idx + 1) % 256 == 0 or idx == len(paths) - 1:
                        app_logger.info(f"Processed {idx + 1} / {len(paths)} images.")
            except Exception as e:
                app_logger.warning(f"Failed to process image {path_str}: {e}")

        embeddings_arr = np.concatenate(embeddings_list, axis=0)

        # Save to database files
        db = EmbeddingDatabase(emb_dir)
        db.save(embeddings_arr, df_valid, {"model_type": model_type})
        
        app_logger.info(f"Embedding generation completed successfully for {len(embeddings_arr)} items.")
    except Exception as e:
        exception_logger.error(f"Embedding generation failed: {e}")
        sys.exit(1)


def train_transfer_learning() -> None:
    """
    Runs category classification classifier fine-tuning.
    """
    app_logger.info("Starting Transfer Learning training...")
    try:
        from training.trainer import TransferLearningTrainer
        trainer = TransferLearningTrainer()
        trainer.run_training()
        app_logger.info("Transfer Learning training completed successfully.")
    except Exception as e:
        exception_logger.error(f"Transfer learning training failed: {e}")
        sys.exit(1)


def train_siamese_network() -> None:
    """
    Trains Siamese network metric learning layers.
    """
    app_logger.info("Starting Siamese Network metric learning training...")
    try:
        from training.siamese_training import SiameseTrainer
        trainer = SiameseTrainer()
        trainer.run_training()
        app_logger.info("Siamese Network training completed successfully.")
    except Exception as e:
        exception_logger.error(f"Siamese training failed: {e}")
        sys.exit(1)


def run_comparative_evaluation() -> None:
    """
    Queries validation images and aggregates retrieval accuracy/speed metrics.
    """
    app_logger.info("Starting System Evaluation...")
    try:
        from recommendation.comparison import SystemComparison
        from recommendation.report_generator import ReportGenerator

        comp = SystemComparison()
        # Evaluate using 30 queries for quicker completion
        results = comp.run_comparative_evaluation(num_queries=30)

        # Save report and generate plots
        rep = ReportGenerator()
        rep.save_comparison_report(results)
        rep.generate_comparison_plots(results)

        app_logger.info("System Evaluation completed successfully.")
    except Exception as e:
        exception_logger.error(f"Evaluation audit failed: {e}")
        sys.exit(1)


def run_benchmarking() -> None:
    """
    Measures latency times and peak memory stats.
    """
    app_logger.info("Starting System Resource Benchmarking...")
    try:
        # Load sample paths
        csv_path = settings.SUBSET_DIRECTORY / "subset_metadata.csv"
        if not csv_path.exists():
            app_logger.error("Subset metadata not found. Run dataset preparation first.")
            sys.exit(1)
        
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        sample_paths = df["image_path"].head(5).tolist()

        from recommendation.benchmark import RecommendationBenchmark
        bench = RecommendationBenchmark()
        
        benchmark_results = {}
        for model in ["baseline", "transfer_learning", "siamese"]:
            # Run benchmark if embeddings exist
            emb_dir = settings.EMBEDDING_DIRECTORY / model
            if (emb_dir / "embeddings.npy").exists():
                summary = bench.run_latency_benchmark(sample_paths, model_type=model, runs=5)
                benchmark_results[model] = summary

        # Save to benchmark report
        import json
        out_path = settings.REPORTS_DIRECTORY / "recommendation_benchmark.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, indent=4)
        
        app_logger.info(f"System Resource Benchmarking completed successfully. Reports saved to {out_path}")
    except Exception as e:
        exception_logger.error(f"Benchmarking execution failed: {e}")
        sys.exit(1)


def main() -> None:
    """
    Parse command line actions.
    """
    parser = argparse.ArgumentParser(description="Image-Based Fashion Product Recommendation CLI Orchestrator")
    parser.add_argument(
        "--action", 
        type=str, 
        required=True,
        choices=[
            "prepare-dataset",
            "validate-dataset",
            "generate-baseline",
            "train-transfer",
            "generate-transfer-embeddings",
            "train-siamese",
            "generate-siamese-embeddings",
            "evaluate",
            "benchmark",
            "all"
        ],
        help="Action operation to execute."
    )

    args = parser.parse_args()

    # Configure TF device allocation on CLI start
    from models.model_utils import configure_accelerator
    configure_accelerator()

    if args.action == "prepare-dataset":
        run_dataset_preparation()
    elif args.action == "validate-dataset":
        run_dataset_validation()
    elif args.action == "generate-baseline":
        generate_embeddings("baseline")
    elif args.action == "train-transfer":
        train_transfer_learning()
    elif args.action == "generate-transfer-embeddings":
        generate_embeddings("transfer_learning")
    elif args.action == "train-siamese":
        train_siamese_network()
    elif args.action == "generate-siamese-embeddings":
        generate_embeddings("siamese")
    elif args.action == "evaluate":
        run_comparative_evaluation()
    elif args.action == "benchmark":
        run_benchmarking()
    elif args.action == "all":
        app_logger.info("=== RUNNING FULL END-TO-END PIPELINE ===")
        run_dataset_preparation()
        run_dataset_validation()
        generate_embeddings("baseline")
        train_transfer_learning()
        generate_embeddings("transfer_learning")
        train_siamese_network()
        generate_embeddings("siamese")
        run_comparative_evaluation()
        run_benchmarking()
        app_logger.info("=== FULL END-TO-END PIPELINE COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
