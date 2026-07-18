"""
Module Description: Streamlit Dataset Explorer Page
Purpose: Renders interactive dataset distribution charts - categories, gender, colour, season, usage breakdowns.
Author: Technical Lead
Version: 2.0.0
Dependencies: streamlit, pandas, pathlib, config.settings
"""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import pandas as pd
import streamlit as st
from config import settings
from app.components.header import render_header
from app.components.footer import render_footer


def load_subset_df() -> pd.DataFrame:
    """Load the balanced subset metadata CSV."""
    csv_path = settings.SUBSET_DIRECTORY / "subset_metadata.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(str(csv_path))
    except Exception:
        return pd.DataFrame()


def render_pie_bar(df: pd.DataFrame, col_name: str, title: str, label_col: str = None) -> None:
    """Renders a horizontal bar chart for a categorical column."""
    if col_name not in df.columns:
        st.warning(f"Column '{col_name}' not found.")
        return
    counts = df[col_name].value_counts().reset_index()
    counts.columns = [col_name, "Count"]
    counts = counts.sort_values("Count", ascending=False)
    st.markdown(
        f'<div style="font-weight:600;color:#e2e8f0;font-size:0.9rem;margin-bottom:0.5rem;">{title}</div>',
        unsafe_allow_html=True
    )
    st.bar_chart(counts.set_index(col_name)["Count"])


def main() -> None:
    """Renders the dataset explorer page."""
    from app.utils.session_manager import SessionManager
    from app.utils.theme import inject_custom_css
    SessionManager.initialize_session_state()
    inject_custom_css()

    render_header("🗄️ Dataset Explorer")

    st.markdown(
        '<p style="color:#94a3b8;font-size:0.92rem;margin-bottom:1.5rem;">'
        'Explore the composition of the balanced fashion product subset used for training, validation, and evaluation.</p>',
        unsafe_allow_html=True
    )

    df = load_subset_df()
    if df.empty:
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:2.5rem;">'
            '<div style="font-size:2.5rem;margin-bottom:0.8rem;">📂</div>'
            '<div style="font-weight:600;color:#e2e8f0;margin-bottom:0.4rem;">Dataset Subset Not Found</div>'
            '<div style="font-size:0.85rem;color:#94a3b8;">'
            'Run <code style="color:#a78bfa;">python main.py --action prepare_dataset</code> to generate the subset.</div>'
            '</div>',
            unsafe_allow_html=True
        )
        render_footer()
        return

    # ── Summary Metrics ────────────────────────────────────────────
    st.markdown('<div class="section-header">📌 Subset Overview</div>', unsafe_allow_html=True)
    col_a, col_b, col_c, col_d = st.columns(4)
    metrics_top = [
        (col_a, str(len(df)),                                           "Total Images"),
        (col_b, str(df["articleType"].nunique()) if "articleType" in df else "N/A",   "Categories"),
        (col_c, str(df["gender"].nunique()) if "gender" in df else "N/A",             "Genders"),
        (col_d, str(df["baseColour"].nunique()) if "baseColour" in df else "N/A",     "Colours"),
    ]
    for col, val, lbl in metrics_top:
        with col:
            st.metric(lbl, val)

    st.markdown("---")

    # ── Tabs for distribution breakdowns ──────────────────────────
    tab_cat, tab_gender, tab_color, tab_season, tab_usage, tab_raw = st.tabs([
        "🛍️ Category",
        "👤 Gender",
        "🎨 Colour",
        "🌤️ Season",
        "🏷️ Usage",
        "📋 Raw Data"
    ])

    with tab_cat:
        st.markdown('<div class="section-header" style="font-size:1rem;">Product Category Distribution</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.82rem;color:#94a3b8;margin-bottom:1rem;">'
            'Count of images per article category in the balanced subset.</p>',
            unsafe_allow_html=True
        )
        col1, col2 = st.columns([2, 1])
        with col1:
            render_pie_bar(df, "articleType", "Image Count by Category")
        with col2:
            if "articleType" in df.columns:
                cat_counts = df["articleType"].value_counts().reset_index()
                cat_counts.columns = ["Category", "Images"]
                cat_counts["% Share"] = (cat_counts["Images"] / len(df) * 100).round(1).astype(str) + "%"
                st.table(cat_counts)

    with tab_gender:
        st.markdown('<div class="section-header" style="font-size:1rem;">Gender Distribution</div>', unsafe_allow_html=True)
        col3, col4 = st.columns([2, 1])
        with col3:
            render_pie_bar(df, "gender", "Image Count by Gender")
        with col4:
            if "gender" in df.columns:
                g_counts = df["gender"].value_counts().reset_index()
                g_counts.columns = ["Gender", "Images"]
                g_counts["% Share"] = (g_counts["Images"] / len(df) * 100).round(1).astype(str) + "%"
                st.table(g_counts)

        # Cross-tab: gender vs category
        if "gender" in df.columns and "articleType" in df.columns:
            st.markdown('<div class="section-header" style="font-size:1rem;">📊 Gender × Category Cross-tab</div>', unsafe_allow_html=True)
            cross = pd.crosstab(df["articleType"], df["gender"])
            st.dataframe(cross, use_container_width=True)

    with tab_color:
        st.markdown('<div class="section-header" style="font-size:1rem;">Colour Distribution (Top 20)</div>', unsafe_allow_html=True)
        if "baseColour" in df.columns:
            top_colors = df["baseColour"].value_counts().head(20).reset_index()
            top_colors.columns = ["Colour", "Count"]
            st.bar_chart(top_colors.set_index("Colour")["Count"])

            col5, col6 = st.columns(2)
            with col5:
                st.markdown('<div style="font-size:0.85rem;color:#94a3b8;margin-bottom:0.4rem;">Top 10 Colours</div>', unsafe_allow_html=True)
                st.table(top_colors.head(10))
            with col6:
                # Colour per category
                if "articleType" in df.columns:
                    st.markdown('<div style="font-size:0.85rem;color:#94a3b8;margin-bottom:0.4rem;">Most Common Colour per Category</div>', unsafe_allow_html=True)
                    top_col_per_cat = (
                        df.groupby("articleType")["baseColour"]
                        .agg(lambda x: x.value_counts().index[0])
                        .reset_index()
                    )
                    top_col_per_cat.columns = ["Category", "Top Colour"]
                    st.table(top_col_per_cat)

    with tab_season:
        st.markdown('<div class="section-header" style="font-size:1rem;">Season Distribution</div>', unsafe_allow_html=True)
        col7, col8 = st.columns([2, 1])
        with col7:
            render_pie_bar(df, "season", "Image Count by Season")
        with col8:
            if "season" in df.columns:
                s_counts = df["season"].value_counts().reset_index()
                s_counts.columns = ["Season", "Images"]
                s_counts["% Share"] = (s_counts["Images"] / len(df) * 100).round(1).astype(str) + "%"
                st.table(s_counts)

        # Year distribution if available
        if "year" in df.columns:
            st.markdown('<div class="section-header" style="font-size:1rem;">📅 Year Distribution</div>', unsafe_allow_html=True)
            year_counts = df["year"].dropna().astype(int).value_counts().sort_index().reset_index()
            year_counts.columns = ["Year", "Images"]
            st.bar_chart(year_counts.set_index("Year")["Images"])

    with tab_usage:
        st.markdown('<div class="section-header" style="font-size:1rem;">Usage Context Distribution</div>', unsafe_allow_html=True)
        col9, col10 = st.columns([2, 1])
        with col9:
            render_pie_bar(df, "usage", "Image Count by Usage")
        with col10:
            if "usage" in df.columns:
                u_counts = df["usage"].value_counts().reset_index()
                u_counts.columns = ["Usage", "Images"]
                u_counts["% Share"] = (u_counts["Images"] / len(df) * 100).round(1).astype(str) + "%"
                st.table(u_counts)

    with tab_raw:
        st.markdown('<div class="section-header" style="font-size:1rem;">📋 Raw Subset Data Sample</div>', unsafe_allow_html=True)

        # Filter controls
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            cat_filter = st.selectbox("Filter by Category", ["All"] + sorted(df["articleType"].unique().tolist()) if "articleType" in df.columns else ["All"])
        with col_f2:
            gender_filter = st.selectbox("Filter by Gender", ["All"] + sorted(df["gender"].unique().tolist()) if "gender" in df.columns else ["All"])
        with col_f3:
            season_filter = st.selectbox("Filter by Season", ["All"] + sorted(df["season"].dropna().unique().tolist()) if "season" in df.columns else ["All"])

        filtered = df.copy()
        if cat_filter != "All" and "articleType" in df.columns:
            filtered = filtered[filtered["articleType"] == cat_filter]
        if gender_filter != "All" and "gender" in df.columns:
            filtered = filtered[filtered["gender"] == gender_filter]
        if season_filter != "All" and "season" in df.columns:
            filtered = filtered[filtered["season"] == season_filter]

        st.markdown(
            f'<div style="font-size:0.82rem;color:#94a3b8;margin-bottom:0.5rem;">'
            f'Showing <b style="color:#a78bfa;">{len(filtered):,}</b> of <b style="color:#60a5fa;">{len(df):,}</b> records</div>',
            unsafe_allow_html=True
        )

        display_cols = [c for c in ["id", "gender", "articleType", "baseColour", "season", "usage", "displayName"] if c in filtered.columns]
        st.dataframe(
            filtered[display_cols].head(200).reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )

    render_footer()


if __name__ == "__main__":
    main()
