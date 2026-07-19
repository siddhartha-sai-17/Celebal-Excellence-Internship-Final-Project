"""
Module Description: Streamlit Visual Search Page
Purpose: Uploads query, filters metadata, computes similarities, maps Grad-CAM, and lists product cards.
Author: Technical Lead
Version: 1.0.0
Dependencies: streamlit, PIL.Image, config.settings, recommendation.recommendation_engine, app.components.sidebar, app.components.upload_widget, app.components.recommendation_grid, app.components.metrics_card, utils.image_utils
"""

import sys
from pathlib import Path
# Ensure the project root is on sys.path regardless of where Streamlit is launched from
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from PIL import Image
from config import settings
from app.components.sidebar import render_sidebar
from app.components.upload_widget import render_upload_widget
from app.components.recommendation_grid import render_recommendation_grid
from app.components.metrics_card import render_metrics_card
from app.components.header import render_header
from app.components.footer import render_footer
from recommendation.recommendation_engine import RecommendationEngine


def main() -> None:
    """Renders the interactive image search interface."""
    from app.utils.session_manager import SessionManager
    from app.utils.theme import inject_custom_css
    SessionManager.initialize_session_state()
    inject_custom_css()

    render_header("🔍 Visually Similar Product Search")

    # Render control sidebar (injects sidebar options to session state)
    render_sidebar()

    # Render uploader widget
    img_pil, img_name = render_upload_widget()

    if img_pil is not None:
        # Store query image in session state
        st.session_state["query_image"] = img_pil
        st.session_state["query_image_path"] = img_name

        # ── Smart category hint from filename ─────────────────────────────────
        # Map filename keywords → the exact articleType values in the dataset
        _KEYWORD_MAP = {
            "watch": "Watches", "watches": "Watches",
            "shirt": "Shirts", "shirts": "Shirts",
            "tshirt": "Tshirts", "t-shirt": "Tshirts", "tee": "Tshirts",
            "jeans": "Jeans", "denim": "Jeans",
            "sandal": "Sandals", "sandals": "Sandals", "slipper": "Sandals",
            "shoe": "Casual Shoes", "shoes": "Casual Shoes", "sneaker": "Casual Shoes",
            "bag": "Handbags", "handbag": "Handbags", "purse": "Handbags",
        }
        current_category_filter = st.session_state.get("filters", {}).get("category", "All")
        hint_applied = st.session_state.get("category_hint_applied_for")

        # If a new image is uploaded and category is not set, auto-select the matching category
        if img_name and hint_applied != img_name:
            name_lower = img_name.lower().replace("_", " ").replace("-", " ")
            suggested_cat = next(
                (cat for kw, cat in _KEYWORD_MAP.items() if kw in name_lower), None
            )
            if suggested_cat:
                st.session_state["filters"]["category"] = suggested_cat
                st.session_state["category_hint_applied_for"] = img_name
                st.rerun()

        # Display confirmation of auto-filtering
        if hint_applied == img_name and current_category_filter != "All":
            st.info(
                f"🎯 **Category filter auto-set to '{current_category_filter}'** "
                f"based on your uploaded filename. Click Generate Recommendations to search."
            )
        # ─────────────────────────────────────────────────────────────────────

        # Trigger search button
        col_run, col_reset = st.columns([1, 4])
        
        with col_run:
            run_clicked = st.button("Generate Recommendations", type="primary")
        with col_reset:
            reset_clicked = st.button("Reset Search", type="secondary")

        if reset_clicked:
            st.session_state["query_image"] = None
            st.session_state["query_image_path"] = None
            st.session_state["recommendations"] = None
            st.session_state["latencies"] = None
            st.session_state["category_hint_applied_for"] = None
            st.session_state["filters"]["category"] = "All"
            st.rerun()


        # Run recommendations
        if run_clicked or st.session_state["recommendations"] is not None:
            # Recompute only on click or if empty
            if run_clicked or st.session_state["recommendations"] is None:
                with st.spinner("Analyzing image features and matching products..."):
                    engine = RecommendationEngine()
                    filters = st.session_state["filters"]
                    
                    # Convert filters "All" to None
                    gender = None if filters.get("gender") == "All" else filters.get("gender")
                    category = None if filters.get("category") == "All" else filters.get("category")
                    color = None if filters.get("color") == "All" else filters.get("color")
                    season = None if filters.get("season") == "All" else filters.get("season")
                    usage = None if filters.get("usage") == "All" else filters.get("usage")

                    recs, latencies = engine.recommend(
                        image_input=st.session_state["query_image"],
                        model_type=st.session_state["selected_model"],
                        gender_filter=gender,
                        category_filter=category,
                        color_filter=color,
                        season_filter=season,
                        usage_filter=usage,
                        similarity_threshold=st.session_state["similarity_threshold"],
                        top_k=st.session_state["top_k"],
                        use_faiss=st.session_state["enable_faiss"]
                    )
                    
                    st.session_state["recommendations"] = recs
                    st.session_state["latencies"] = latencies

            # Render latencies and metrics cards
            if st.session_state["latencies"] is not None:
                render_metrics_card(st.session_state["latencies"], st.session_state["enable_faiss"])

            # Render recommendations cards
            recs = st.session_state["recommendations"]
            if recs is not None:
                if recs:
                    render_recommendation_grid(recs)

                    # If top results span multiple categories, show a filter tip
                    top_cats = [r.get("category", "") for r in recs[:5]]
                    unique_cats = set(top_cats)
                    if len(unique_cats) > 1 or (len(recs) >= 3 and len(unique_cats) == 1 and
                                                 list(unique_cats)[0].lower() not in ["watches", "watch"]):
                        st.info(
                            "💡 **Tip — Getting wrong category?** "
                            "Use the **Category** filter in the left sidebar to restrict results "
                            "to a specific product type (e.g. 'Watches', 'Shirts', 'Handbags'). "
                            "The engine will automatically find the best matches within that category."
                        )
                else:
                    st.warning("⚠️ **No visually similar products found matching the criteria.**")
                    
                    # ── Debugger block for zero results ───────────────────────────
                    with st.expander("🛠️ Diagnostics: Why are there 0 results?", expanded=True):
                        try:
                            import numpy as np
                            from recommendation.embedding_selector import EmbeddingSelector
                            from recommendation.cache_manager import CacheManager
                            from preprocessing.image_preprocessor import ImagePreprocessor
                            
                            model_name = st.session_state["selected_model"]
                            emb_dir, checkpoint_path = EmbeddingSelector.get_paths(model_name)
                            cache = CacheManager()
                            model = cache.get_model(model_name, checkpoint_path)
                            db_embs, db_meta = cache.get_database(model_name, emb_dir)
                            
                            img_tensor = ImagePreprocessor.preprocess_for_inference(st.session_state["query_image"])
                            
                            st.write(f"**Model selected:** `{model_name}`")
                            st.write(f"**Checkpoint path resolved:** `{checkpoint_path}`")
                            st.write(f"**Model load status:** `{'Success' if model is not None else 'Failed'}`")
                            st.write(f"**Database embeddings found:** `{'Yes' if db_embs is not None else 'No'}` (shape: `{db_embs.shape if db_embs is not None else 'N/A'}`)")
                            st.write(f"**Database metadata rows:** `{len(db_meta) if db_meta is not None else 'N/A'}`")
                            
                            if model is not None and img_tensor is not None:
                                emb = model(img_tensor, training=False).numpy().squeeze()
                                st.write(f"**Query Embedding Shape:** `{emb.shape}`")
                                st.write(f"**Query Embedding contains NaNs:** `{np.isnan(emb).any()}`")
                                st.write(f"**Query Embedding Min/Max/Mean:** `{np.min(emb):.4f}` / `{np.max(emb):.4f}` / `{np.mean(emb):.4f}`")
                        except Exception as debug_err:
                            st.error(f"Diagnostics runner error: {debug_err}")
                    # ──────────────────────────────────────────────────────────────


            # Render Explainability (Grad-CAM Activation Heatmaps)
            st.markdown("---")
            st.markdown("### 🧠 Explainable AI: Activation Maps (Grad-CAM)")

            with st.expander("Why were these products recommended?"):
                st.write(
                    "Grad-CAM visualizes which regional features in your uploaded image "
                    "(such as collars, sleeves, dials, strap buckles, or heels) "
                    "contributed most to the generated embedding representation."
                )
                # Gate behind a button so heavy model load only runs on demand
                if st.button("🔍 Generate Activation Heatmap", key="gradcam_btn"):
                    with st.spinner("Generating attention heatmaps..."):
                        try:
                            from recommendation.embedding_selector import EmbeddingSelector
                            from recommendation.cache_manager import CacheManager
                            from preprocessing.image_preprocessor import ImagePreprocessor
                            from utils.image_utils import generate_gradcam, overlay_heatmap_on_image

                            _, checkpoint_path = EmbeddingSelector.get_paths(st.session_state["selected_model"])
                            cache = CacheManager()
                            model = cache.get_model(st.session_state["selected_model"], checkpoint_path)

                            img_tensor = ImagePreprocessor.preprocess_for_inference(st.session_state["query_image"])

                            if model and img_tensor is not None:
                                heatmap = generate_gradcam(model, img_tensor)

                                if heatmap is not None:
                                    temp_query_path = settings.SUBSET_DIRECTORY / "temp_query_gradcam.jpg"
                                    st.session_state["query_image"].save(temp_query_path)
                                    superimposed_img = overlay_heatmap_on_image(temp_query_path, heatmap)

                                    col_orig, col_grad = st.columns(2)
                                    with col_orig:
                                        st.image(st.session_state["query_image"], caption="Original Query", use_column_width=True)
                                    with col_grad:
                                        st.image(superimposed_img, caption="Activation Focus Overlay", use_column_width=True)

                                    if temp_query_path.exists():
                                        temp_query_path.unlink()
                                else:
                                    st.warning("Grad-CAM heatmap calculation returned empty. Convolution layer auto-detection failed.")
                        except Exception as e:
                            st.warning(f"Unable to render Grad-CAM explainability maps: {e}")


    else:
        # Prompt search uploader start
        st.info("💡 **Awaiting Query Image**: Please upload or drag-and-drop a product image to initiate similar search queries.")

    render_footer()


if __name__ == "__main__":
    main()
