"""
Module Description: UI Header Component
Purpose: Renders a consistent heading block and subheadings at the top of each page.
Author: Technical Lead
Version: 2.0.0
Dependencies: streamlit
"""

import streamlit as st
from config import settings


def render_header(page_title: str) -> None:
    """
    Renders a premium gradient page header with version badge.

    Args:
        page_title: Title of the active page.
    """
    st.markdown(
        f'<div class="custom-header fade-in">{page_title}</div>'
        f'<div class="custom-subheader">'
        f'{settings.PROJECT_NAME} &nbsp;'
        f'<span class="tech-tag">v{settings.PROJECT_VERSION}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

