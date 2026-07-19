"""
Module Description: Streamlit Home Page
Purpose: Renders project objectives, target dataset subset statistics, and tech stack details.
Author: Technical Lead
Version: 2.0.0
Dependencies: streamlit, pathlib, config.settings
"""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import streamlit as st
from config import settings
from app.components.header import render_header
from app.components.footer import render_footer



def _render_content_cloud_mode() -> None:
    """Renders the home page content using only pre-computed evaluation reports
    (no raw dataset required). Used when running in Streamlit Cloud without Kaggle creds."""
    import json

    # Objective cards
    st.markdown('<div class="section-header">🎯 Project Objectives</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#94a3b8;font-size:0.92rem;line-height:1.7;margin-bottom:1rem;">'
        'Textual keyword search fails to capture stylistic visual similarity — cut, texture, pattern, and colour combinations. '
        'This platform addresses that gap with an end-to-end <b style="color:#a78bfa;">visual search &amp; product recommendation</b> '
        'system powered by deep neural embeddings.</p>',
        unsafe_allow_html=True
    )
    c1, c2, c3 = st.columns(3)
    methods = [
        (c1, "🧠", "Baseline CNN",      "#a78bfa", "Pretrained ResNet50 (ImageNet) with a custom L2-normalised projection head."),
        (c2, "🔬", "Transfer Learning", "#818cf8", "Category classifier fine-tuned on balanced fashion subset with staged unfreezing."),
        (c3, "⚡", "Siamese Network",   "#60a5fa", "Contrastive metric learning — optimises pairwise embedding distances for superior visual similarity."),
    ]
    for col, icon, title, color, desc in methods:
        with col:
            st.markdown(
                f'<div class="glass-card" style="text-align:center;">'
                f'<div style="font-size:2rem;margin-bottom:0.4rem;">{icon}</div>'
                f'<div style="font-weight:700;color:{color};margin-bottom:0.5rem;">{title}</div>'
                f'<div style="font-size:0.8rem;color:#94a3b8;line-height:1.6;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # Pre-computed analytics
    st.markdown('<div class="section-header">📊 Dataset Analytics</div>', unsafe_allow_html=True)
    stats_file = settings.EVALUATION_DIRECTORY / "dataset_statistics.json"
    val_report_file = settings.EVALUATION_DIRECTORY / "dataset_validation_report.json"
    subset_size, images_per_cat, report = 0, {}, {}

    if stats_file.exists():
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                stats = json.load(f)
                subset_size = stats.get("total_images", 0)
                images_per_cat = stats.get("images_per_category", {})
        except Exception:
            pass

    if val_report_file.exists():
        try:
            with open(val_report_file, "r", encoding="utf-8") as f:
                report = json.load(f)
        except Exception:
            pass

    st.markdown('<span class="status-ok">● Pre-computed embeddings loaded</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    metrics_data = [
        (m1, f"{report.get('total_metadata_records', 'N/A')}", "Raw Records"),
        (m2, f"{report.get('total_images_in_folder', 'N/A')}", "Images on Disk"),
        (m3, f"{subset_size:,}" if subset_size else "1,750", "Subset Size"),
        (m4, f"{len(report.get('category_distribution', {})) or 43}", "Categories"),
    ]
    for col, val, lbl in metrics_data:
        with col:
            st.metric(lbl, val)

    if images_per_cat:
        import pandas as pd
        st.markdown('<div class="section-header" style="font-size:1rem;">🛍️ Category Distribution</div>', unsafe_allow_html=True)
        dist_df = pd.DataFrame(list(images_per_cat.items()), columns=["Category", "Image Count"])
        dist_df = dist_df.sort_values("Image Count", ascending=False).reset_index(drop=True)
        st.table(dist_df)

    # Tech stack
    st.markdown('<div class="section-header">💻 Technology Stack</div>', unsafe_allow_html=True)
    tags = [
        "TensorFlow 2.x", "Keras 3", "ResNet50", "EfficientNetB0",
        "FAISS", "Streamlit", "NumPy", "Pandas", "OpenCV", "Scikit-Learn", "Python 3.11"
    ]
    st.markdown(
        " ".join(f'<span class="tech-tag">{t}</span>' for t in tags),
        unsafe_allow_html=True
    )


def main() -> None:
    """Renders the home page dashboard."""
    from app.utils.session_manager import SessionManager
    from app.utils.theme import inject_custom_css
    SessionManager.initialize_session_state()
    inject_custom_css()

    render_header("🏠 Project Overview")

    raw_styles_csv = settings.DATASET_DIRECTORY / "styles.csv"
    raw_images_dir = settings.DATASET_DIRECTORY / "images"
    cloud_mode = not raw_styles_csv.exists() or not raw_images_dir.exists()

    import os
    kaggle_configured = (
        ("KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ) or
        (hasattr(st, "secrets") and "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets)
    )

    if cloud_mode:
        if kaggle_configured:
            from utils.kaggle_downloader import download_fashion_dataset, dataset_is_present
            st.info(
                "⬇️ **Dataset not found locally.** Attempting to download from Kaggle using credentials. "
                "This happens only once and may take 1–3 minutes depending on your connection."
            )
            with st.spinner("📦 Downloading Fashion Product Images dataset from Kaggle..."):
                success = download_fashion_dataset(settings.DATASET_DIRECTORY)
            if success and dataset_is_present(raw_images_dir, raw_styles_csv):
                st.success("✅ Dataset downloaded successfully! Reloading page...")
                st.rerun()
            else:
                st.warning(
                    "⚠️ **Kaggle download could not complete.** Please check that your "
                    "`KAGGLE_USERNAME` and `KAGGLE_KEY` secrets are valid."
                )
        else:
            st.success(
                "🚀 **Cloud Mode Active**: Pre-compiled model embeddings and 1,750 subset "
                "product images are loaded directly from the repository."
            )
        
        _render_content_cloud_mode()
        render_footer()
        return
    else:

        from preprocessing.dataset_validator import DatasetValidator
        from preprocessing.dataset_subset import DatasetSubsetGenerator
        validator = DatasetValidator(raw_styles_csv, raw_images_dir)
        report = validator.run_validation()
        validator.save_report()
        subset_gen = DatasetSubsetGenerator()
        subset_gen.generate_subset()



    # ── Objective cards ────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Project Objectives</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#94a3b8;font-size:0.92rem;line-height:1.7;margin-bottom:1rem;">'
        'Textual keyword search fails to capture stylistic visual similarity — cut, texture, pattern, and colour combinations. '
        'This platform addresses that gap with an end-to-end <b style="color:#a78bfa;">visual search &amp; product recommendation</b> '
        'system powered by deep neural embeddings.</p>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    methods = [
        (c1, "🧠", "Baseline CNN",       "#a78bfa", "Pretrained ResNet50 (ImageNet) with a custom L2-normalised projection head."),
        (c2, "🔬", "Transfer Learning",  "#818cf8", "Category classifier fine-tuned on balanced fashion subset with staged unfreezing."),
        (c3, "⚡", "Siamese Network",    "#60a5fa", "Contrastive metric learning — optimises pairwise embedding distances for superior visual similarity."),
    ]
    for col, icon, title, color, desc in methods:
        with col:
            st.markdown(
                f'<div class="glass-card" style="text-align:center;">'
                f'<div style="font-size:2rem;margin-bottom:0.4rem;">{icon}</div>'
                f'<div style="font-weight:700;color:{color};margin-bottom:0.5rem;">{title}</div>'
                f'<div style="font-size:0.8rem;color:#94a3b8;line-height:1.6;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── Dataset Analytics ──────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Dataset Analytics</div>', unsafe_allow_html=True)

    # Load pre-computed stats from evaluation reports (works in both local and cloud modes)
    stats_file = settings.EVALUATION_DIRECTORY / "dataset_statistics.json"
    val_report_file = settings.EVALUATION_DIRECTORY / "dataset_validation_report.json"
    subset_size, images_per_cat, report = 0, {}, {}

    if stats_file.exists():
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                stats = json.load(f)
                subset_size = stats.get("total_images", 0)
                images_per_cat = stats.get("images_per_category", {})
        except Exception:
            pass

    if val_report_file.exists():
        try:
            with open(val_report_file, "r", encoding="utf-8") as f:
                report = json.load(f)
        except Exception:
            pass

    if not cloud_mode and report.get("metadata_exists") and report.get("images_dir_exists"):
        st.markdown('<span class="status-ok">● Dataset validated</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-ok">● Pre-computed embeddings loaded</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    metrics_data = [
        (m1, f"{report.get('total_metadata_records', 'N/A')}", "Raw Records"),
        (m2, f"{report.get('total_images_in_folder', 'N/A')}", "Images on Disk"),
        (m3, f"{subset_size:,}" if subset_size else "1,750", "Subset Size"),
        (m4, f"{len(report.get('category_distribution', {})) or 43}", "Categories"),
    ]
    for col, val, lbl in metrics_data:
        with col:
            st.metric(lbl, val)

    if images_per_cat:
        import pandas as pd
        st.markdown('<div class="section-header" style="font-size:1rem;">🛍️ Category Distribution</div>', unsafe_allow_html=True)
        dist_df = pd.DataFrame(list(images_per_cat.items()), columns=["Category", "Image Count"])
        dist_df = dist_df.sort_values("Image Count", ascending=False).reset_index(drop=True)
        st.table(dist_df)


    # ── Tech stack ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">💻 Technology Stack</div>', unsafe_allow_html=True)
    tags = [
        "TensorFlow 2.x", "Keras 3", "ResNet50", "EfficientNetB0",
        "FAISS", "Streamlit", "NumPy", "Pandas", "OpenCV", "Scikit-Learn", "Python 3.11"
    ]
    st.markdown(
        " ".join(f'<span class="tech-tag">{t}</span>' for t in tags),
        unsafe_allow_html=True
    )

    render_footer()


if __name__ == "__main__":
    import pandas as pd
    main()
