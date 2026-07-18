"""
Module Description: Streamlit Frontend Unit Tests
Purpose: Assures session managers correctly initialize state parameters and reset query fields.
Author: Technical Lead
Version: 1.0.0
Dependencies: pytest, app.utils.session_manager
"""

import pytest
import streamlit as st
from app.utils.session_manager import SessionManager


def test_session_manager_initialization() -> None:
    """Ensures session states keys populate on initialize call."""
    # Call initialize
    SessionManager.initialize_session_state()

    assert "selected_model" in st.session_state
    assert "top_k" in st.session_state
    assert "similarity_threshold" in st.session_state
    assert "recommendations" in st.session_state
    assert "filters" in st.session_state

    # Set dummy query state
    st.session_state["query_image"] = "dummy_pil"
    st.session_state["recommendations"] = [{"rank": 1}]

    # Reset
    SessionManager.reset_search()
    assert st.session_state["query_image"] is None
    assert st.session_state["recommendations"] is None
