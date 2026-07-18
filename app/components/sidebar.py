"""
Module Description: UI Sidebar Component
Purpose: Renders global options: model selection, top-k slider, FAISS flag, and dropdown metadata filters.
Author: Technical Lead
Version: 2.0.0
Dependencies: streamlit, config.settings, recommendation.metadata_loader
"""

import streamlit as st
from config import settings
from recommendation.metadata_loader import MetadataLoader


def render_sidebar() -> None:
    """
    Draws premium sidebar controls and updates corresponding session state keys.
    """
    with st.sidebar:
        # Branded sidebar header
        st.markdown(
            '<div style="text-align:center;padding:0.5rem 0 1rem;">'
            '<div style="font-size:1.4rem;font-weight:800;background:linear-gradient(135deg,#a78bfa,#60a5fa);'
            '-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">🛍️ Controls</div>'
            '<div style="font-size:0.72rem;color:rgba(148,163,184,0.5);margin-top:0.2rem;">Configure search parameters</div>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")

        # 1. Model selection with icons
        st.markdown('<div style="font-size:0.75rem;font-weight:600;color:#a78bfa;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;">Embedding Model</div>', unsafe_allow_html=True)
        model_options = {
            "🧠 Baseline CNN": "baseline",
            "🔬 Transfer Learning": "transfer_learning",
            "⚡ Siamese Network": "siamese"
        }
        selected_label = st.selectbox(
            "Model",
            options=list(model_options.keys()),
            index=list(model_options.values()).index(st.session_state["selected_model"]),
            label_visibility="collapsed"
        )
        st.session_state["selected_model"] = model_options[selected_label]

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Top-K results slider
        st.markdown('<div style="font-size:0.75rem;font-weight:600;color:#a78bfa;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;">Top-K Results</div>', unsafe_allow_html=True)
        st.session_state["top_k"] = st.slider(
            "Top-K",
            min_value=5,
            max_value=50,
            value=st.session_state["top_k"],
            step=5,
            label_visibility="collapsed"
        )

        # 3. Similarity threshold slider
        st.markdown('<div style="font-size:0.75rem;font-weight:600;color:#a78bfa;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;">Similarity Threshold</div>', unsafe_allow_html=True)
        st.session_state["similarity_threshold"] = st.slider(
            "Similarity",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state["similarity_threshold"],
            step=0.05,
            label_visibility="collapsed"
        )

        # 4. FAISS toggle
        st.markdown("<br>", unsafe_allow_html=True)
        st.session_state["enable_faiss"] = st.toggle(
            "⚡ FAISS Acceleration",
            value=st.session_state["enable_faiss"]
        )

        st.markdown("---")

        # 5. Category Filters section
        st.markdown(
            '<div style="font-size:0.75rem;font-weight:600;color:#818cf8;text-transform:uppercase;'
            'letter-spacing:0.08em;margin-bottom:0.8rem;">🔍 Filters</div>',
            unsafe_allow_html=True
        )

        csv_path = settings.SUBSET_DIRECTORY / "subset_metadata.csv"
        loader = MetadataLoader(csv_path)
        filter_opts = loader.get_filter_options()

        def render_dropdown_filter(label: str, key: str, options_list: list) -> None:
            opts = ["All"] + options_list
            current_val = st.session_state["filters"].get(key, "All")
            index = opts.index(current_val) if current_val in opts else 0
            selected = st.selectbox(label, options=opts, index=index)
            st.session_state["filters"][key] = selected

        render_dropdown_filter("Gender", "gender", filter_opts["genders"])
        render_dropdown_filter("Category", "category", filter_opts["categories"])
        render_dropdown_filter("Color", "color", filter_opts["colors"])
        render_dropdown_filter("Season", "season", filter_opts["seasons"])
        render_dropdown_filter("Usage", "usage", filter_opts["usages"])

        st.markdown("---")
        if st.button("🔄 Reset All Settings", type="secondary", use_container_width=True):
            st.session_state["query_image"] = None
            st.session_state["query_image_path"] = None
            st.session_state["recommendations"] = None
            st.session_state["latencies"] = None
            st.session_state["filters"] = {
                "gender": "All", "category": "All", "color": "All",
                "season": "All", "usage": "All"
            }
            from recommendation.cache_manager import CacheManager
            CacheManager().clear_all_caches()
            st.rerun()
