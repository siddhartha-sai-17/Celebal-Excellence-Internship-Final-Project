"""
Module Description: Metrics Card Component
Purpose: Renders elapsed latency metrics and performance stats in premium glassmorphism tiles.
Author: Technical Lead
Version: 2.0.0
Dependencies: streamlit
"""

import streamlit as st
from typing import Dict


def render_metrics_card(latencies: Dict[str, float], enable_faiss: bool) -> None:
    """
    Displays processing speed, search latency, and indexing indicator as premium tiles.

    Args:
        latencies: Dictionary containing latency metrics.
        enable_faiss: Toggle state of FAISS indexing.
    """
    st.markdown('<div class="section-header">⚡ Performance Metrics</div>', unsafe_allow_html=True)

    total  = latencies.get("total_elapsed", 0.0) * 1000
    inf    = latencies.get("inference", 0.0) * 1000
    sim    = latencies.get("similarity_search", 0.0) * 1000
    engine = "FAISS Index" if enable_faiss else "Cosine"

    col1, col2, col3, col4 = st.columns(4)
    tiles = [
        (col1, f"{total:.1f} ms",  "Total Latency",      "#a78bfa"),
        (col2, f"{inf:.1f} ms",    "CNN Inference",       "#818cf8"),
        (col3, f"{sim:.1f} ms",    "Similarity Search",   "#60a5fa"),
        (col4, engine,             "Match Engine",        "#34d399"),
    ]
    for col, value, label, color in tiles:
        with col:
            st.markdown(
                f'<div class="metric-tile">'
                f'<span class="value" style="background:linear-gradient(135deg,{color},{color}99);'
                f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
                f'background-clip:text;">{value}</span>'
                f'<span class="label">{label}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    with st.expander("📊 Full Latency Breakdown"):
        rows = [
            ("Image Preprocessing",       latencies.get("preprocessing", 0.0)),
            ("CNN Feature Extraction",     latencies.get("inference", 0.0)),
            ("Similarity Distance Search", latencies.get("similarity_search", 0.0)),
            ("Sorting & Metadata Filter",  latencies.get("ranking", 0.0)),
            ("Total System Elapsed",       latencies.get("total_elapsed", 0.0)),
        ]
        for label, val_s in rows:
            pct = min((val_s / max(latencies.get("total_elapsed", 1e-9), 1e-9)) * 100, 100)
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'margin-bottom:0.5rem;">'
                f'<span style="font-size:0.82rem;color:#94a3b8;">{label}</span>'
                f'<span style="font-size:0.82rem;font-weight:600;color:#a78bfa;">{val_s*1000:.1f} ms</span>'
                f'</div>'
                f'<div style="height:4px;background:rgba(255,255,255,0.06);border-radius:2px;margin-bottom:0.8rem;">'
                f'<div style="height:4px;width:{pct:.1f}%;background:linear-gradient(90deg,#6366f1,#8b5cf6);'
                f'border-radius:2px;"></div></div>',
                unsafe_allow_html=True
            )
