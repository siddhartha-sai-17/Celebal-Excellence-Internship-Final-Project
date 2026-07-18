"""
Module Description: Recommendation Results Grid Component
Purpose: Renders ranked visually similar products in a premium dark glassmorphism card grid.
Author: Technical Lead
Version: 2.0.0
Dependencies: streamlit, pathlib, PIL.Image, config.settings, utils.logger
"""

import streamlit as st
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
from config import settings
from utils.logger import app_logger


def render_recommendation_grid(recommendations: List[Dict[str, Any]]) -> None:
    """
    Lays out product results inside a premium responsive glassmorphism card grid.

    Args:
        recommendations: Ranked product recommendation list.
    """
    if not recommendations:
        st.warning("No visually similar products found matching the criteria.")
        return

    st.markdown(
        '<div class="section-header">🎯 Recommended Products</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div style="color:rgba(148,163,184,0.6);font-size:0.85rem;margin-bottom:1rem;">'
        f'Showing <b style="color:#a78bfa;">{len(recommendations)}</b> visually similar matches</div>',
        unsafe_allow_html=True
    )

    cols_per_row = 3
    rows_count = (len(recommendations) + cols_per_row - 1) // cols_per_row

    for row_idx in range(rows_count):
        cols = st.columns(cols_per_row, gap="medium")
        for col_idx in range(cols_per_row):
            item_idx = row_idx * cols_per_row + col_idx
            if item_idx < len(recommendations):
                item = recommendations[item_idx]
                col = cols[col_idx]

                with col:
                    # Similarity colour: green → yellow → red
                    sim_pct = item.get("similarity_score", 0.0)
                    if sim_pct >= 0.75:
                        sim_color = "#34d399"
                    elif sim_pct >= 0.50:
                        sim_color = "#fbbf24"
                    else:
                        sim_color = "#f87171"

                    # Resolve product image — check committed subset images first (200x200),
                    # then fall back to the full local dataset folder.
                    img_id = str(item.get("image_id", ""))
                    img_name = f"{img_id}.jpg" if img_id else ""
                    raw_path = Path(item.get("image_path", ""))
                    if raw_path and not raw_path.is_absolute():
                        raw_path = settings.BASE_DIR / raw_path

                    # Priority: committed subset images → full raw dataset
                    subset_img = settings.SUBSET_DIRECTORY / "images" / img_name
                    if img_name and subset_img.exists():
                        img_path = subset_img
                    elif raw_path and raw_path.exists():
                        img_path = raw_path
                    else:
                        img_path = None

                    if img_path and img_path.exists():
                        try:
                            st.image(str(img_path), use_column_width=True)
                        except Exception as e:
                            app_logger.error(f"Failed to display card image {img_path}: {e}")
                            st.markdown('<div class="skeleton-box"></div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="skeleton-box"></div>', unsafe_allow_html=True)

                    # Card metadata below the image
                    card_html = f"""
                    <div style="
                        background: rgba(255,255,255,0.04);
                        border: 1px solid rgba(255,255,255,0.08);
                        border-radius: 12px;
                        padding: 0.9rem;
                        margin-top: 0.4rem;
                        transition: border-color 0.2s;
                    ">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
                            <span style="
                                background: linear-gradient(135deg,#6366f1,#8b5cf6);
                                color: white; font-weight: 700; font-size: 0.7rem;
                                padding: 2px 8px; border-radius: 10px;
                            ">#{item['rank']}</span>
                            <span style="
                                color: {sim_color}; font-weight: 700; font-size: 0.78rem;
                                background: rgba(255,255,255,0.05);
                                border: 1px solid {sim_color}44;
                                padding: 2px 8px; border-radius: 10px;
                            ">{item['similarity_percentage']} match</span>
                        </div>
                        <div style="font-weight:600;font-size:0.88rem;color:#e2e8f0;margin-bottom:0.4rem;
                                    overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
                             title="{item['product_name']}">
                            {item['product_name']}
                        </div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.2rem;">
                            <div style="font-size:0.72rem;color:rgba(148,163,184,0.7);">
                                <span style="color:#818cf8;">Cat</span> {item['category']}
                            </div>
                            <div style="font-size:0.72rem;color:rgba(148,163,184,0.7);">
                                <span style="color:#818cf8;">Color</span> {item['color']}
                            </div>
                            <div style="font-size:0.72rem;color:rgba(148,163,184,0.7);">
                                <span style="color:#818cf8;">Sub</span> {item['subcategory']}
                            </div>
                            <div style="font-size:0.72rem;color:rgba(148,163,184,0.7);">
                                <span style="color:#818cf8;">Season</span> {item['season']}
                            </div>
                        </div>
                    </div>
                    """
                    col.markdown(card_html, unsafe_allow_html=True)
