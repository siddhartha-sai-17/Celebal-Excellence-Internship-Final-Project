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
                    # Force reload modules to bypass Streamlit Cloud stale memory caches
                    import importlib
                    import recommendation.ranking_engine
                    import recommendation.recommendation_service
                    import recommendation.recommendation_engine
                    importlib.reload(recommendation.ranking_engine)
                    importlib.reload(recommendation.recommendation_service)
                    importlib.reload(recommendation.recommendation_engine)
                    
                    from recommendation.recommendation_engine import RecommendationEngine
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
                    
                    # Safely load trace from session state or static attribute
                    trace_list = st.session_state.get("search_trace")
                    if not trace_list:
                        from recommendation.ranking_engine import RankingEngine
                        trace_list = getattr(RankingEngine, "last_trace", ["(No class-level trace found)"])
                    trace_text = "\n".join(trace_list)
                    
                    st.info(
                        f"⚙️ **Main Search Debugger:**\n\n"
                        f"Found **{len(recs)}** results.\n\n"
                        f"**Parameters sent to recommender:**\n"
                        f"gender={gender}, category={category}, color={color}, season={season}, usage={usage}\n\n"
                        f"**Internal Ranking Trace:**\n"
                        f"```\n"
                        f"{trace_text}\n"
                        f"```"
                    )

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
                            from recommendation.similarity_engine import SimilarityEngine
                            
                            model_name = st.session_state["selected_model"]
                            emb_dir, checkpoint_path = EmbeddingSelector.get_paths(model_name)
                            cache = CacheManager()
                            model = cache.get_model(model_name, checkpoint_path)
                            db_embs, db_meta = cache.get_database(model_name, emb_dir)
                            img_tensor = ImagePreprocessor.preprocess_for_inference(st.session_state["query_image"])
                            
                            lines = []
                            lines.append(f"Model Selected: {model_name}")
                            lines.append(f"Checkpoint Path: {checkpoint_path}")
                            lines.append(f"Model Load Status: {'SUCCESS' if model is not None else 'FAILED'}")
                            lines.append(f"Database Embeddings: {'FOUND' if db_embs is not None else 'NOT FOUND'}")
                            if db_embs is not None:
                                lines.append(f"  Shape: {db_embs.shape}")
                            lines.append(f"Database Metadata: {'FOUND' if db_meta is not None else 'NOT FOUND'}")
                            if db_meta is not None:
                                lines.append(f"  Rows count: {len(db_meta)}")
                                lines.append(f"  Columns: {list(db_meta.columns)}")
                            
                            if model is not None and img_tensor is not None and db_embs is not None and db_meta is not None:
                                # Extract query embedding
                                q_emb = model(img_tensor, training=False).numpy().squeeze()
                                lines.append(f"Query Embedding Shape: {q_emb.shape}")
                                q_nan = np.isnan(q_emb).any()
                                lines.append(f"Query Embedding contains NaNs: {q_nan}")
                                if not q_nan:
                                    lines.append(f"Query Embedding Min/Max/Mean: {np.min(q_emb):.4f} / {np.max(q_emb):.4f} / {np.mean(q_emb):.4f}")
                                
                                # Compute similarity
                                scores = SimilarityEngine.compute_similarity(q_emb, db_embs, metric=settings.SIMILARITY_ALGORITHM)
                                lines.append(f"Similarity Scores count: {len(scores)}")
                                s_nan = np.isnan(scores).any()
                                lines.append(f"Similarity Scores contain NaNs: {s_nan}")
                                if not s_nan:
                                    lines.append(f"Similarity Scores Min/Max/Mean: {np.min(scores):.4f} / {np.max(scores):.4f} / {np.mean(scores):.4f}")
                                
                                # Category checks
                                current_cat = st.session_state["filters"].get("category", "All")
                                current_gender = st.session_state["filters"].get("gender", "All")
                                current_color = st.session_state["filters"].get("color", "All")
                                current_season = st.session_state["filters"].get("season", "All")
                                current_usage = st.session_state["filters"].get("usage", "All")
                                
                                lines.append(f"Active Filters — Cat: {current_cat}, Gen: {current_gender}, Col: {current_color}, Sea: {current_season}, Usg: {current_usage}")
                                
                                # Step-by-step trace of RankingEngine.rank_and_filter logic
                                scored_trace = []
                                match_cat_count = 0
                                match_gender_count = 0
                                match_color_count = 0
                                match_season_count = 0
                                match_usage_count = 0
                                
                                for idx, row in db_meta.iterrows():
                                    # Category filter
                                    is_cat_match = True
                                    if current_cat != "All":
                                        is_cat_match = str(row.get("articleType", "")).lower() == current_cat.lower()
                                    if is_cat_match:
                                        match_cat_count += 1
                                        
                                    # Gender filter
                                    is_gen_match = True
                                    if current_gender != "All":
                                        is_gen_match = str(row.get("gender", "")).lower() == current_gender.lower()
                                    if is_cat_match and is_gen_match:
                                        match_gender_count += 1
                                        
                                    # Color filter
                                    is_col_match = True
                                    if current_color != "All":
                                        is_col_match = str(row.get("baseColour", "")).lower() == current_color.lower()
                                    if is_cat_match and is_gen_match and is_col_match:
                                        match_color_count += 1
                                        
                                    # Season filter
                                    is_sea_match = True
                                    if current_season != "All":
                                        is_sea_match = str(row.get("season", "")).lower() == current_season.lower()
                                    if is_cat_match and is_gen_match and is_col_match and is_sea_match:
                                        match_season_count += 1
                                        
                                    # Usage filter
                                    is_usg_match = True
                                    if current_usage != "All":
                                        is_usg_match = str(row.get("usage", "")).lower() == current_usage.lower()
                                    if is_cat_match and is_gen_match and is_col_match and is_sea_match and is_usg_match:
                                        match_usage_count += 1
                                        scored_trace.append((float(scores[idx]), str(row["id"]), row))
                                
                                lines.append(f"Trace Counters:")
                                lines.append(f"  Rows matching Category filter: {match_cat_count}")
                                lines.append(f"  Rows matching Cat + Gender: {match_gender_count}")
                                lines.append(f"  Rows matching Cat + Gen + Color: {match_color_count}")
                                lines.append(f"  Rows matching Cat + Gen + Col + Season: {match_season_count}")
                                lines.append(f"  Rows matching All Filters (scored size): {len(scored_trace)}")
                                
                                if scored_trace:
                                    scored_trace.sort(key=lambda x: x[0], reverse=True)
                                    lines.append(f"  Top 3 scored after filters: {scored_trace[0][0]:.4f}, {scored_trace[1][0]:.4f}, {scored_trace[2][0]:.4f}" if len(scored_trace) >= 3 else f"  Scored count: {len(scored_trace)}")
                                    
                                    # Threshold check
                                    thresh = st.session_state["similarity_threshold"]
                                    filtered_count = sum(1 for s, _, _ in scored_trace if s >= thresh)
                                    lines.append(f"  Rows passing similarity threshold ({thresh}): {filtered_count}")
                                    
                            st.code("\n".join(lines))

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
