"""
Module Description: Streamlit Model Comparison Page
Purpose: Renders recommendations for a single query across Baseline, Transfer, and Siamese models side-by-side.
Author: Technical Lead
Version: 2.0.0
Dependencies: streamlit, recommendation.recommendation_engine
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
    """Renders the model comparison dashboard."""
    from app.utils.session_manager import SessionManager
    from app.utils.theme import inject_custom_css
    SessionManager.initialize_session_state()
    inject_custom_css()

    render_header("🔬 Model Comparison")

    st.markdown(
        '<p style="color:#94a3b8;font-size:0.92rem;margin-bottom:1.5rem;">'
        'Compare visual recommendations retrieved by all three embedding engines for the same query image.</p>',
        unsafe_allow_html=True
    )

    query_image = st.session_state.get("query_image")

    if query_image is None:
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:2.5rem;">'
            '<div style="font-size:2.5rem;margin-bottom:0.8rem;">🖼️</div>'
            '<div style="font-weight:600;color:#e2e8f0;margin-bottom:0.4rem;">No Query Image Selected</div>'
            '<div style="font-size:0.85rem;color:#94a3b8;">Upload an image on the <b style="color:#a78bfa;">Image Search</b> page first, then return here to compare models.</div>'
            '</div>',
            unsafe_allow_html=True
        )
        render_footer()
        return

    # Show query image
    st.markdown('<div class="section-header">📷 Query Image</div>', unsafe_allow_html=True)
    col_q, col_info = st.columns([1, 3])
    with col_q:
        st.image(query_image, use_column_width=True)
    with col_info:
        st.markdown(
            '<div class="glass-card">'
            '<div style="font-size:0.78rem;color:#818cf8;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.6rem;">Running comparison across</div>'
            '<div style="display:flex;gap:0.5rem;flex-wrap:wrap;">'
            '<span class="tech-tag">🧠 Baseline CNN</span>'
            '<span class="tech-tag">🔬 Transfer Learning</span>'
            '<span class="tech-tag">⚡ Siamese Network</span>'
            '</div>'
            '<div style="font-size:0.8rem;color:#94a3b8;margin-top:0.8rem;">Each model uses its own learned embedding space. '
            'Siamese typically yields the tightest semantic matches.</div>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Run all three models
    from recommendation.recommendation_engine import RecommendationEngine

    models = [
        ("🧠 Baseline CNN",      "baseline",         "#a78bfa"),
        ("🔬 Transfer Learning", "transfer_learning", "#818cf8"),
        ("⚡ Siamese Network",   "siamese",           "#60a5fa"),
    ]

    top_k    = st.session_state.get("top_k", 10)
    threshold = st.session_state.get("similarity_threshold", 0.0)
    use_faiss = st.session_state.get("enable_faiss", True)

    cols = st.columns(3, gap="medium")
    engine = RecommendationEngine()

    for (col, (label, model_type, color)) in zip(cols, models):
        with col:
            st.markdown(
                f'<div style="font-weight:700;color:{color};font-size:1rem;'
                f'border-left:3px solid {color};padding-left:0.6rem;margin-bottom:0.8rem;">'
                f'{label}</div>',
                unsafe_allow_html=True
            )
            with st.spinner(f"Loading {label}..."):
                try:
                    recs, lats = engine.recommend(
                        image_input=query_image,
                        model_type=model_type,
                        top_k=min(top_k, 6),
                        similarity_threshold=threshold,
                        use_faiss=use_faiss
                    )
                    if recs:
                        # Latency pill
                        total_ms = lats.get("total_elapsed", 0.0) * 1000
                        st.markdown(
                            f'<div style="font-size:0.72rem;color:#94a3b8;margin-bottom:0.6rem;">'
                            f'<span style="background:rgba(99,102,241,0.15);color:#a78bfa;'
                            f'padding:2px 8px;border-radius:8px;font-weight:600;">'
                            f'{total_ms:.0f} ms</span> &nbsp; {len(recs)} results</div>',
                            unsafe_allow_html=True
                        )
                        # Top 6 images
                        for item in recs[:6]:
                            img_path = Path(item["image_path"])
                            if not img_path.is_absolute():
                                img_path = settings.BASE_DIR / img_path
                            if img_path.exists():
                                st.image(str(img_path), use_column_width=True)
                                st.markdown(
                                    f'<div style="font-size:0.72rem;color:#94a3b8;text-align:center;'
                                    f'margin-bottom:0.4rem;">'
                                    f'<span style="color:{color};font-weight:600;">'
                                    f'{item["similarity_percentage"]}</span> · {item["product_name"][:22]}…'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                    else:
                        st.info("No matches found.")
                except Exception as e:
                    st.warning(f"Could not load {label}: {e}")

    render_footer()


if __name__ == "__main__":
    main()
