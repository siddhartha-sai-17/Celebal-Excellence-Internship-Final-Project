"""
Module Description: Visualization and Plotting Utilities
Purpose: Generates pie charts, bar charts, training curves, similarity distributions, pair/triplet previews, PCA/t-SNE projections, and Grad-CAM explainability plots.
Author: Technical Lead
Version: 1.0.0
Dependencies: matplotlib, seaborn, numpy, sklearn.decomposition, sklearn.manifold, config.settings, utils.logger
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from config import settings
from utils.logger import app_logger


def plot_category_distribution(df: pd.DataFrame, output_dir: Path = settings.PLOTS_DIRECTORY) -> None:
    """
    Generates and saves bar and pie charts for category distribution.

    Args:
        df: DataFrame containing the dataset with a column 'articleType'.
        output_dir: Directory where the plots will be saved.
    """
    if "articleType" not in df.columns:
        app_logger.warning("No 'articleType' column found for category distribution plotting.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = df["articleType"].value_counts()

    # Define color palette
    colors = sns.color_palette("muted", len(counts))

    # 1. Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis", hue=counts.index, legend=False)
    plt.title("Category Distribution (Subset)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "category_bar_chart.png", dpi=150)
    plt.close()

    # 2. Pie Chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        textprops={'fontsize': 10}
    )
    plt.title("Category Distribution Percentage", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_dir / "category_pie_chart.png", dpi=150)
    plt.close()

    app_logger.info(f"Category distribution plots saved to {output_dir}")


def plot_training_curves(history: Dict[str, List[float]], 
                         output_name: str,
                         output_dir: Path = settings.PLOTS_DIRECTORY) -> None:
    """
    Plots and saves loss and accuracy curves for training history.

    Args:
        history: Dictionary containing training and validation loss/accuracy lists.
        output_name: Base filename prefix for saved plots (e.g. 'transfer_learning_curves').
        output_dir: Target directory for plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(next(iter(history.values()))) + 1)

    # 1. Loss Curve
    if "loss" in history:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["loss"], "b-", label="Training Loss")
        if "val_loss" in history:
            plt.plot(epochs, history["val_loss"], "r-", label="Validation Loss")
        plt.title("Loss Curves", fontsize=12, fontweight="bold")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_dir / f"{output_name}_loss.png", dpi=150)
        plt.close()

    # 2. Accuracy Curve
    if "accuracy" in history or "acc" in history:
        acc_key = "accuracy" if "accuracy" in history else "acc"
        val_acc_key = "val_accuracy" if "val_accuracy" in history else "val_acc"
        
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history[acc_key], "b-", label="Training Accuracy")
        if val_acc_key in history:
            plt.plot(epochs, history[val_acc_key], "r-", label="Validation Accuracy")
        plt.title("Accuracy Curves", fontsize=12, fontweight="bold")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_dir / f"{output_name}_accuracy.png", dpi=150)
        plt.close()

    app_logger.info(f"Training curves saved to {output_dir}")


def plot_embeddings_space(embeddings: np.ndarray, 
                          labels: List[str], 
                          output_name: str,
                          method: str = "pca", 
                          output_dir: Path = settings.PLOTS_DIRECTORY) -> None:
    """
    Performs dimensionality reduction and plots the embeddings space labeled by category.

    Args:
        embeddings: Numpy array of shape (N, embedding_dim).
        labels: List of category label names (len N).
        output_name: Filename prefix for the plot.
        method: Projection method, either "pca" or "tsne".
        output_dir: Target directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    num_samples = len(embeddings)
    if num_samples < 5:
        app_logger.warning("Not enough samples to plot embedding projections.")
        return

    # Projection
    if method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=settings.RANDOM_SEED)
        coords = reducer.fit_transform(embeddings)
        title = "Embedding Space PCA Projection"
    elif method.lower() == "tsne":
        # Adjust perplexity for small datasets
        perplexity = min(30, num_samples - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=settings.RANDOM_SEED)
        coords = reducer.fit_transform(embeddings)
        title = "Embedding Space t-SNE Projection"
    else:
        app_logger.error(f"Unsupported reduction method: {method}")
        return

    plt.figure(figsize=(10, 8))
    df_proj = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "Category": labels
    })
    
    sns.scatterplot(data=df_proj, x="x", y="y", hue="Category", palette="tab10", alpha=0.8, s=60)
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / f"{output_name}_{method}.png", dpi=150)
    plt.close()

    app_logger.info(f"Embeddings projection ({method}) saved as {output_name} to {output_dir}")


def plot_similarity_distribution(similarity_matrix: np.ndarray,
                                 output_name: str,
                                 output_dir: Path = settings.PLOTS_DIRECTORY) -> None:
    """
    Plots a histogram distribution of pairwise similarity values.

    Args:
        similarity_matrix: A 2D numpy array representing similarity metrics.
        output_name: File save name.
        output_dir: Folder to save plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Flatten upper triangle of similarity matrix
    tri_upper = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    
    plt.figure(figsize=(8, 5))
    sns.histplot(tri_upper, kde=True, color="teal", bins=30)
    plt.title("Pairwise Similarity Distribution", fontsize=12, fontweight="bold")
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / f"{output_name}_similarity_dist.png", dpi=150)
    plt.close()

    app_logger.info(f"Similarity distribution plot saved to {output_dir}")
