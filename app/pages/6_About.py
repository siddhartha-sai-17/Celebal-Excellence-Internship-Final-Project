"""
Module Description: Streamlit About Page
Purpose: Renders directory layout, configuration parameters, and system credits.
Author: Technical Lead
Version: 2.0.0
Dependencies: streamlit, config.settings
"""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from config import settings
from app.components.header import render_header
from app.components.footer import render_footer


def main() -> None:
    """Renders the about page."""
    from app.utils.session_manager import SessionManager
    from app.utils.theme import inject_custom_css
    SessionManager.initialize_session_state()
    inject_custom_css()

    render_header("ℹ️ About This System")

    st.markdown(
        '<p style="color:#94a3b8;font-size:0.92rem;line-height:1.7;margin-bottom:1.5rem;">'
        'An end-to-end deep learning pipeline for image-based fashion product recommendation, '
        'combining CNN feature extraction, transfer learning, and Siamese metric learning.</p>',
        unsafe_allow_html=True
    )

    # ── Architecture overview ──────────────────────────────────────
    st.markdown('<div class="section-header">🏗️ System Architecture</div>', unsafe_allow_html=True)
    phases = [
        ("1", "Dataset Preparation",    "Balanced 1,750-image subset across 7 fashion categories with validation auditing."),
        ("2", "Image Preprocessing",    "Resize → normalize → augment pipeline using OpenCV and Keras ImageDataGenerator."),
        ("3", "Feature Extraction",     "Baseline ResNet50 embeddings (2048-D) with L2-normalised projection head."),
        ("4", "Transfer Learning",      "Category-supervised fine-tuning with staged backbone unfreezing."),
        ("5", "Siamese Network",        "Contrastive loss training on positive/negative image pairs for metric embedding."),
        ("6", "FAISS Indexing",         "Flat inner-product index for sub-millisecond approximate nearest-neighbour search."),
        ("7", "Streamlit Interface",    "Multi-page app with Grad-CAM explainability, real-time metrics, and model comparison."),
    ]
    for num, title, desc in phases:
        st.markdown(
            f'<div style="display:flex;gap:1rem;margin-bottom:0.7rem;align-items:flex-start;">'
            f'<span style="background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;font-weight:700;'
            f'border-radius:50%;width:30px;height:30px;display:flex;align-items:center;'
            f'justify-content:center;font-size:0.78rem;flex-shrink:0;">{num}</span>'
            f'<div>'
            f'<div style="font-weight:600;color:#e2e8f0;font-size:0.9rem;">{title}</div>'
            f'<div style="font-size:0.8rem;color:#94a3b8;margin-top:0.2rem;">{desc}</div>'
            f'</div></div>',
            unsafe_allow_html=True
        )

    # ── Configuration ──────────────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ Active Configuration</div>', unsafe_allow_html=True)
    config_items = [
        ("Project",            settings.PROJECT_NAME),
        ("Version",            settings.PROJECT_VERSION),
        ("Target Categories",  ", ".join(getattr(settings, "TARGET_CATEGORIES", []))),
        ("Subset Size",        str(getattr(settings, "SUBSET_SIZE_PER_CATEGORY", "N/A"))),
        ("Image Size",         f"{getattr(settings, 'IMAGE_SIZE', ('?','?'))[0]}×{getattr(settings, 'IMAGE_SIZE', ('?','?'))[1]}"),
        ("Embedding Dim",      str(getattr(settings, "EMBEDDING_DIM", "N/A"))),
        ("Base Directory",     str(settings.BASE_DIR)),
    ]
    c1, c2 = st.columns(2)
    for i, (key, val) in enumerate(config_items):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:0.5rem 0.8rem;margin-bottom:0.4rem;'
                f'background:rgba(255,255,255,0.03);border-radius:8px;'
                f'border:1px solid rgba(255,255,255,0.06);">'
                f'<span style="font-size:0.78rem;color:#818cf8;font-weight:600;">{key}</span>'
                f'<span style="font-size:0.78rem;color:#cbd5e1;text-align:right;max-width:60%;'
                f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{val}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── Directory layout ───────────────────────────────────────────
    st.markdown('<div class="section-header">📁 Directory Structure</div>', unsafe_allow_html=True)
    tree = """
```
Vision Product Recommendation/
├── app/               # Streamlit UI (pages, components, utils)
├── config/            # settings.py — global configuration
├── dataset/           # Raw Fashion Product Images dataset
├── data/subset/       # Balanced image subset + metadata CSV
├── models/            # Saved model checkpoints
├── embeddings/        # Pre-computed .npy embedding matrices
├── faiss_index/       # FAISS flat index files
├── evaluation/        # Benchmark reports & dataset statistics
├── preprocessing/     # Dataset validation, subset, image pipeline
├── training/          # Transfer learning & Siamese training scripts
├── siamese/           # Siamese dataset, model, contrastive loss
├── recommendation/    # Engine, FAISS search, metadata loader, cache
├── optimization/      # System validator, FAISS builder
├── utils/             # Logger, timer, image utils
└── tests/             # 27 unit tests (all passing)
```
    """
    st.markdown(tree)

    # ── Tech stack ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">🛠️ Technology Stack</div>', unsafe_allow_html=True)
    stack = [
        ("Deep Learning",   ["TensorFlow 2.x", "Keras 3", "ResNet50", "EfficientNetB0"]),
        ("Search & Index",  ["FAISS", "NumPy", "Scikit-Learn"]),
        ("Data & Vision",   ["OpenCV", "Pillow", "Pandas"]),
        ("Interface",       ["Streamlit", "Plotly", "Matplotlib"]),
        ("Language",        ["Python 3.11"]),
    ]
    for category, tools in stack:
        st.markdown(
            f'<div style="margin-bottom:0.6rem;">'
            f'<span style="font-size:0.75rem;color:#818cf8;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.07em;">{category}: </span>'
            + " ".join(f'<span class="tech-tag">{t}</span>' for t in tools)
            + '</div>',
            unsafe_allow_html=True
        )

    render_footer()


if __name__ == "__main__":
    main()
