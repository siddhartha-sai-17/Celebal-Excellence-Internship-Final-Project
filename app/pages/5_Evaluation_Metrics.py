"""
Module Description: Streamlit Model Evaluation & Metrics Page
Purpose: Renders retrieval metrics (Precision@K, Recall@K, mAP, MRR, NDCG) and training history loss/accuracy graphs.
Author: Technical Lead
Version: 2.0.0
Dependencies: streamlit, pandas, matplotlib, json, pathlib, config.settings
"""

import sys
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import settings
from app.components.header import render_header
from app.components.footer import render_footer


def find_latest_checkpoint_dir(prefix: str) -> Path:
    """Finds the latest checkpoint directory matching the given prefix."""
    checkpoints_dir = settings.BASE_DIR / "models" / "checkpoints"
    if not checkpoints_dir.exists():
        return None
    dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if not dirs:
        return None
    dirs.sort()
    return dirs[-1]


def load_json(filepath: Path) -> dict:
    """Loads a JSON file safely."""
    if not filepath or not filepath.exists():
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def main() -> None:
    """Renders the evaluation metrics and graphs dashboard."""
    from app.utils.session_manager import SessionManager
    from app.utils.theme import inject_custom_css
    SessionManager.initialize_session_state()
    inject_custom_css()

    render_header("📊 Model Evaluation & Metrics")

    st.markdown(
        '<p style="color:#94a3b8;font-size:0.92rem;margin-bottom:1.5rem;">'
        'Analyze retrieval accuracy metrics, rank-aware evaluations, and model training loss curves side-by-side.</p>',
        unsafe_allow_html=True
    )

    # ── Load comparison report data ────────────────────────────────
    comparison_file = settings.EVALUATION_DIRECTORY / "reports" / "comparison_report.json"
    comparison_data = load_json(comparison_file)

    if not comparison_data:
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:2.5rem;">'
            '<div style="font-size:2.5rem;margin-bottom:0.8rem;">📊</div>'
            '<div style="font-weight:600;color:#e2e8f0;margin-bottom:0.4rem;">Evaluation Report Not Found</div>'
            '<div style="font-size:0.85rem;color:#94a3b8;">'
            'Please run the evaluation command to generate metrics: '
            '<code style="color:#a78bfa;">python main.py --action benchmark</code>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )
        render_footer()
        return

    # Tabs for different metrics categories
    tab_acc, tab_rank, tab_train = st.tabs([
        "🎯 Retrieval Accuracy",
        "🏆 Rank & Retrieval Quality",
        "📈 Model Training Logs"
    ])

    # Model names mapping
    model_labels = {
        "baseline": "Baseline CNN",
        "transfer_learning": "Transfer Learning",
        "siamese": "Siamese Network"
    }

    # ── Tab 1: Retrieval Accuracy ──────────────────────────────────
    with tab_acc:
        st.markdown('<div class="section-header">🎯 Accuracy Metrics (Precision & Recall @ K)</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.85rem;color:#94a3b8;margin-bottom:1rem;">'
            '<b>Precision@K</b> measures the proportion of recommended items that belong to the same product subcategory. '
            '<b>Recall@K</b> measures the proportion of all relevant items from the subcategory successfully retrieved.</p>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        # Precision@K Chart
        with col1:
            st.markdown('<div style="font-weight:600;color:#e2e8f0;font-size:0.9rem;margin-bottom:0.5rem;">Precision @ K</div>', unsafe_allow_html=True)
            
            p_data = []
            for m_key, m_val in comparison_data.items():
                m_label = model_labels.get(m_key, m_key)
                for k in [5, 10, 20]:
                    p_data.append({
                        "Model": m_label,
                        "K": f"K={k}",
                        "Precision": m_val.get(f"precision_at_{k}", 0.0) * 100
                    })
            p_df = pd.DataFrame(p_data)
            
            # Pivot for Streamlit Native Bar Chart
            p_pivot = p_df.pivot(index="K", columns="Model", values="Precision")
            st.bar_chart(p_pivot)

        # Recall@K Chart
        with col2:
            st.markdown('<div style="font-weight:600;color:#e2e8f0;font-size:0.9rem;margin-bottom:0.5rem;">Recall @ K (%)</div>', unsafe_allow_html=True)
            
            r_data = []
            for m_key, m_val in comparison_data.items():
                m_label = model_labels.get(m_key, m_key)
                for k in [5, 10, 20]:
                    r_data.append({
                        "Model": m_label,
                        "K": f"K={k}",
                        "Recall": m_val.get(f"recall_at_{k}", 0.0) * 100
                    })
            r_df = pd.DataFrame(r_data)
            r_pivot = r_df.pivot(index="K", columns="Model", values="Recall")
            st.bar_chart(r_pivot)

        # Mean Average Precision (mAP) Comparison
        st.markdown('<div class="section-header" style="font-size:1rem;">📌 Mean Average Precision (mAP)</div>', unsafe_allow_html=True)
        ap_data = []
        for m_key, m_val in comparison_data.items():
            m_label = model_labels.get(m_key, m_key)
            ap_data.append({
                "Model": m_label,
                "mAP @ 5": f"{m_val.get('ap_at_5', 0.0):.3f}",
                "mAP @ 10": f"{m_val.get('ap_at_10', 0.0):.3f}",
                "mAP @ 20": f"{m_val.get('ap_at_20', 0.0):.3f}"
            })
        st.table(pd.DataFrame(ap_data))

    # ── Tab 2: Rank & Retrieval Quality ───────────────────────────
    with tab_rank:
        st.markdown('<div class="section-header">🏆 Rank-Aware Retrieval Quality</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.85rem;color:#94a3b8;margin-bottom:1rem;">'
            '<b>NDCG@K</b> (Normalized Discounted Cumulative Gain) heavily weights highly-relevant matches placed near the top. '
            '<b>MRR@K</b> (Mean Reciprocal Rank) measures the rank of the first relevant result. '
            '<b>Hit Rate@K</b> measures the probability that at least one relevant product is returned in the top-K.</p>',
            unsafe_allow_html=True
        )

        col3, col4 = st.columns(2)

        # NDCG@K Chart
        with col3:
            st.markdown('<div style="font-weight:600;color:#e2e8f0;font-size:0.9rem;margin-bottom:0.5rem;">NDCG @ K</div>', unsafe_allow_html=True)
            ndcg_data = []
            for m_key, m_val in comparison_data.items():
                m_label = model_labels.get(m_key, m_key)
                for k in [5, 10, 20]:
                    ndcg_data.append({
                        "Model": m_label,
                        "K": f"K={k}",
                        "NDCG": m_val.get(f"ndcg_at_{k}", 0.0)
                    })
            n_df = pd.DataFrame(ndcg_data)
            n_pivot = n_df.pivot(index="K", columns="Model", values="NDCG")
            st.bar_chart(n_pivot)

        # Hit Rate@K Chart
        with col4:
            st.markdown('<div style="font-weight:600;color:#e2e8f0;font-size:0.9rem;margin-bottom:0.5rem;">Hit Rate @ K</div>', unsafe_allow_html=True)
            hr_data = []
            for m_key, m_val in comparison_data.items():
                m_label = model_labels.get(m_key, m_key)
                for k in [5, 10, 20]:
                    hr_data.append({
                        "Model": m_label,
                        "K": f"K={k}",
                        "Hit Rate": m_val.get(f"hit_rate_at_{k}", 0.0)
                    })
            h_df = pd.DataFrame(hr_data)
            h_pivot = h_df.pivot(index="K", columns="Model", values="Hit Rate")
            st.bar_chart(h_pivot)

        # Summary of Rank-Aware metrics in a table
        st.markdown('<div class="section-header" style="font-size:1rem;">📌 Reciprocal Rank Comparison</div>', unsafe_allow_html=True)
        mrr_data = []
        for m_key, m_val in comparison_data.items():
            m_label = model_labels.get(m_key, m_key)
            mrr_data.append({
                "Model": m_label,
                "MRR @ 5": f"{m_val.get('mrr_at_5', 0.0):.3f}",
                "MRR @ 10": f"{m_val.get('mrr_at_10', 0.0):.3f}",
                "MRR @ 20": f"{m_val.get('mrr_at_20', 0.0):.3f}"
            })
        st.table(pd.DataFrame(mrr_data))

    # ── Tab 3: Model Training Logs ─────────────────────────────────
    with tab_train:
        st.markdown('<div class="section-header">📈 Neural Network Training Performance</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.85rem;color:#94a3b8;margin-bottom:1rem;">'
            'Visualizes training loss curves extracted directly from the checkpoints of the fine-tuned and Siamese neural models.</p>',
            unsafe_allow_html=True
        )

        # Locate latest checkpoints
        transfer_dir = find_latest_checkpoint_dir("resnet50_transfer_v1_")
        siamese_dir = find_latest_checkpoint_dir("resnet50_siamese_v1_")

        transfer_history = load_json(transfer_dir / "history.json") if transfer_dir else {}
        siamese_history = load_json(siamese_dir / "history.json") if siamese_dir else {}

        if not transfer_history and not siamese_history:
            st.info("No training history records found in checkpoints.")
            render_footer()
            return

        plt.style.use("dark_background")

        # Transfer Learning curves
        if transfer_history:
            st.markdown('<div class="section-header" style="font-size:1.1rem; border-left-color: #818cf8;">🔬 Transfer Learning Classification Training</div>', unsafe_allow_html=True)
            col_tl_loss, col_tl_acc = st.columns(2)

            epochs_tl = list(range(1, len(transfer_history.get("loss", [])) + 1))
            
            with col_tl_loss:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(epochs_tl, transfer_history.get("loss", []), label="Train Loss", marker='o', color="#818cf8", linewidth=2)
                ax.plot(epochs_tl, transfer_history.get("val_loss", []), label="Val Loss", marker='x', color="#fbbf24", linestyle='--', linewidth=2)
                ax.set_title("Cross-Entropy Loss History", fontsize=10, pad=10, color="#e2e8f0")
                ax.set_xlabel("Epochs", fontsize=8, color="#94a3b8")
                ax.set_ylabel("Loss", fontsize=8, color="#94a3b8")
                ax.grid(True, color="#ffffff", alpha=0.06, linestyle="-")
                ax.set_xticks(epochs_tl)
                ax.legend(frameon=True, facecolor=(0, 0, 0, 0.5), edgecolor=(1, 1, 1, 0.1))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                st.pyplot(fig)

            with col_tl_acc:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(epochs_tl, [a * 100 for a in transfer_history.get("accuracy", [])], label="Train Accuracy", marker='o', color="#34d399", linewidth=2)
                ax.plot(epochs_tl, [a * 100 for a in transfer_history.get("val_accuracy", [])], label="Val Accuracy", marker='x', color="#fbbf24", linestyle='--', linewidth=2)
                ax.set_title("Classification Accuracy History (%)", fontsize=10, pad=10, color="#e2e8f0")
                ax.set_xlabel("Epochs", fontsize=8, color="#94a3b8")
                ax.set_ylabel("Accuracy (%)", fontsize=8, color="#94a3b8")
                ax.grid(True, color="#ffffff", alpha=0.06, linestyle="-")
                ax.set_xticks(epochs_tl)
                ax.legend(frameon=True, facecolor=(0, 0, 0, 0.5), edgecolor=(1, 1, 1, 0.1))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                st.pyplot(fig)

        # Siamese curves
        if siamese_history:
            st.markdown('<div class="section-header" style="font-size:1.1rem; border-left-color: #60a5fa;">⚡ Siamese Metric Learning Training</div>', unsafe_allow_html=True)
            col_s_loss, col_s_stats = st.columns([2, 1])

            epochs_s = list(range(1, len(siamese_history.get("loss", [])) + 1))

            with col_s_loss:
                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.plot(epochs_s, siamese_history.get("loss", []), label="Train Loss", marker='o', color="#60a5fa", linewidth=2)
                ax.plot(epochs_s, siamese_history.get("val_loss", []), label="Val Loss", marker='x', color="#f87171", linestyle='--', linewidth=2)
                ax.set_title("Margin Contrastive Loss History", fontsize=10, pad=10, color="#e2e8f0")
                ax.set_xlabel("Epochs", fontsize=8, color="#94a3b8")
                ax.set_ylabel("Contrastive Loss", fontsize=8, color="#94a3b8")
                ax.grid(True, color="#ffffff", alpha=0.06, linestyle="-")
                ax.set_xticks(epochs_s)
                ax.legend(frameon=True, facecolor=(0, 0, 0, 0.5), edgecolor=(1, 1, 1, 0.1))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                st.pyplot(fig)

            with col_s_stats:
                # Load metadata
                siamese_meta = load_json(siamese_dir / "metadata.json") if siamese_dir else {}
                
                st.markdown('<div style="font-weight:600;color:#e2e8f0;font-size:0.9rem;margin-bottom:0.8rem;">Siamese Metadata</div>', unsafe_allow_html=True)
                metrics = [
                    ("Model Type", siamese_meta.get("model_type", "Siamese (Functional)")),
                    ("Margin Parameter", str(siamese_meta.get("margin", "0.5"))),
                    ("Preloaded RAM Cache", "Enabled (✓)"),
                    ("Optimiser", "Adam"),
                    ("Backbone state", "Frozen (22x Speedup)"),
                ]
                for key, val in metrics:
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;padding:0.4rem 0;'
                        f'border-bottom:1px solid rgba(255,255,255,0.05);font-size:0.78rem;">'
                        f'<span style="color:#94a3b8;">{key}</span>'
                        f'<span style="font-weight:600;color:#60a5fa;">{val}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    render_footer()


if __name__ == "__main__":
    main()
