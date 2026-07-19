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
    # Split prefix emoji and title if present
    parts = page_title.split(" ", 1)
    if len(parts) == 2 and len(parts[0]) <= 2:  # likely an emoji
        emoji, title = parts[0], parts[1]
        header_html = (
            f'<div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.3rem;" class="fade-in">'
            f'<span style="font-size: 2.2rem; line-height: 1;">{emoji}</span>'
            f'<span class="custom-header" style="margin: 0; padding: 0; line-height: 1.2;">{title}</span>'
            f'</div>'
        )
    else:
        header_html = f'<div class="custom-header fade-in">{page_title}</div>'

    st.markdown(
        f'{header_html}'
        f'<div class="custom-subheader">'
        f'{settings.PROJECT_NAME} &nbsp;'
        f'<span class="tech-tag">v{settings.PROJECT_VERSION}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

