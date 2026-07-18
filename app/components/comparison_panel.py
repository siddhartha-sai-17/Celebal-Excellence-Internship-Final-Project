"""
Module Description: Model Comparison Panel Component
Purpose: Renders recommendations side-by-side across Baseline, Transfer Learning, and Siamese configurations in premium dark glassmorphism style.
Author: Technical Lead
Version: 2.0.0
Dependencies: streamlit, pathlib, PIL.Image, config.settings
"""

import streamlit as st
from pathlib import Path
from typing import Dict, Any, List
from config import settings


def render_comparison_panel(baseline_recs: List[Dict[str, Any]],
                            transfer_recs: List[Dict[str, Any]],
                            siamese_recs: List[Dict[str, Any]],
                            limit: int = 5) -> None:
    """
    Lays out side-by-side columns comparing retrieved visual recommendations.

    Args:
        baseline_recs: Baseline results list.
        transfer_recs: Transfer learning results list.
        siamese_recs: Siamese results list.
        limit: Max products to show per column.
    """
    st.markdown("### 📊 Side-by-Side Model Comparison")
    
    col1, col2, col3 = st.columns(3)

    # Helper function to render a product column
    def render_model_column(column_obj: Any, title: str, recs: List[Dict[str, Any]], count_limit: int, color: str) -> None:
        with column_obj:
            st.markdown(
                f'<div style="font-weight:700;color:{color};font-size:1rem;'
                f'border-left:3px solid {color};padding-left:0.6rem;margin-bottom:0.8rem;">'
                f'{title}</div>',
                unsafe_allow_html=True
            )
            st.markdown("---")
            
            if not recs:
                st.write("No matching results found.")
                return

            for idx, item in enumerate(recs[:count_limit]):
                # Determine similarity color
                sim_pct = item.get("similarity_score", 0.0)
                if sim_pct >= 0.75:
                    sim_color = "#34d399"
                elif sim_pct >= 0.50:
                    sim_color = "#fbbf24"
                else:
                    sim_color = "#f87171"

                # Render item card
                item_card_html = f"""
                <div style="
                    background: rgba(255,255,255,0.04);
                    border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 12px;
                    padding: 0.9rem;
                    margin-bottom: 0.8rem;
                    transition: border-color 0.2s;
                ">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
                        <span style="background: linear-gradient(135deg,#6366f1,#8b5cf6);color:white;font-weight:700;font-size:0.7rem;padding:2px 8px;border-radius:10px;">#{idx+1}</span>
                        <span style="color: {sim_color};font-weight:700;font-size:0.78rem;background:rgba(255,255,255,0.05);border:1px solid {sim_color}44;padding:2px 8px;border-radius:10px;">{item['similarity_percentage']} match</span>
                    </div>
                    <div style="font-weight:600;font-size:0.88rem;color:#e2e8f0;margin-bottom:0.4rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="{item['product_name']}">{item['product_name']}</div>
                    <div style="font-size:0.72rem;color:rgba(148,163,184,0.7);"><span style="color:#818cf8;">Cat</span> {item['category']} | <span style="color:#818cf8;">Color</span> {item['color']}</div>
                </div>
                """
                st.markdown(item_card_html, unsafe_allow_html=True)
                
                img_path = Path(item["image_path"])
                if not img_path.is_absolute():
                    img_path = settings.BASE_DIR / img_path
                if img_path.exists():
                    st.image(str(img_path), use_column_width=True)
                else:
                    st.markdown('<div class="skeleton-box"></div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)

    render_model_column(col1, "🧠 Baseline CNN", baseline_recs, limit, "#a78bfa")
    render_model_column(col2, "🔬 Transfer Learning", transfer_recs, limit, "#818cf8")
    render_model_column(col3, "⚡ Siamese Network", siamese_recs, limit, "#60a5fa")
