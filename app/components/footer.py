"""
Module Description: UI Footer Component
Purpose: Renders a premium footer block with version, copyright, and tech credits.
Author: Technical Lead
Version: 2.0.0
Dependencies: streamlit
"""

import streamlit as st
from config import settings


def render_footer() -> None:
    """
    Appends a premium unified footer element at the bottom of the Streamlit layout.
    """
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align:center;padding:1rem 0 0.5rem;">
            <div style="font-size:0.8rem;color:rgba(148,163,184,0.5);line-height:1.8;">
                <span style="color:#a78bfa;font-weight:600;">{settings.PROJECT_NAME}</span>
                &nbsp;·&nbsp; v{settings.PROJECT_VERSION}
                &nbsp;·&nbsp; Python 3.11 &nbsp;·&nbsp; TensorFlow &amp; Keras
            </div>
            <div style="font-size:0.75rem;color:rgba(148,163,184,0.3);margin-top:0.3rem;">
                © 2026 Image-Based Fashion Search. All Rights Reserved.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
