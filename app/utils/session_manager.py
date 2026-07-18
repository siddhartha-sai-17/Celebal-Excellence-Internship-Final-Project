"""
Module Description: Streamlit Session State Manager
Purpose: Standardizes initialization and persistence of user query states across multiple pages.
Author: Technical Lead
Version: 1.0.0
Dependencies: streamlit
"""

import streamlit as st
from config import settings


class SessionManager:
    """
    Manages session state variables for Streamlit presentation layer.
    """

    @staticmethod
    def initialize_session_state() -> None:
        """
        Declares default properties if they do not exist.
        """
        defaults = {
            "query_image": None,          # PIL Image or path
            "query_image_path": None,     # str path
            "selected_model": settings.EMBEDDING_SOURCE,
            "top_k": settings.TOP_K_RESULTS,
            "similarity_threshold": settings.SIMILARITY_THRESHOLD,
            "recommendations": None,      # List[Dict[str, Any]]
            "latencies": None,            # Dict[str, float]
            "filters": {
                "gender": "All",
                "category": "All",
                "color": "All",
                "season": "All",
                "usage": "All"
            },
            "enable_faiss": settings.ENABLE_FAISS
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def reset_search() -> None:
        """Resets the query state variables back to defaults."""
        st.session_state["query_image"] = None
        st.session_state["query_image_path"] = None
        st.session_state["recommendations"] = None
        st.session_state["latencies"] = None
        st.session_state["filters"] = {
            "gender": "All",
            "category": "All",
            "color": "All",
            "season": "All",
            "usage": "All"
        }
