"""
Module Description: Streamlit Entry Point
Purpose: Initializes session managers, validates paths and assets, and launches the presentation layer.
Author: Technical Lead
Version: 1.0.0
Dependencies: streamlit, config.settings, app.utils.session_manager, app.utils.theme, optimization.validator
"""

import sys
from pathlib import Path
# Ensure the project root is on sys.path regardless of where Streamlit is launched from
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from config import settings
from app.utils.session_manager import SessionManager
from app.utils.theme import inject_custom_css
from optimization.validator import SystemValidator

# Configure page layout (wide, custom tab details)
st.set_page_config(
    page_title=settings.PROJECT_NAME,
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Initialize session variables
SessionManager.initialize_session_state()

# 2. Inject global custom styling (shadows, border radiuses, loading skeletons)
inject_custom_css()

# 3. Run validation checks
validator = SystemValidator()
val_status = validator.run_validation_checks()

# Renders premium hero banner
st.markdown("""
<div class="hero-banner fade-in">
    <div class="hero-title">🛍️ Fashion Visual Search</div>
    <div class="hero-subtitle">Deep learning–powered image similarity & product recommendation system</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="section-header">🚀 Getting Started</div>', unsafe_allow_html=True)
    st.markdown("""
    This platform implements **visual similarity search** across fashion apparel, shoes, handbags, watches, and more.
    It benchmarks three deep embedding architectures:
    """)

    st.markdown("""
    <div class="glass-card">
        <b style="color:#a78bfa;">① Baseline CNN</b><br>
        <span style="color:#94a3b8;font-size:0.9rem;">Pretrained ResNet50 (ImageNet weights) with a custom L2-normalized projection head.</span>
    </div>
    <div class="glass-card">
        <b style="color:#818cf8;">② Transfer Learning</b><br>
        <span style="color:#94a3b8;font-size:0.9rem;">Category classifier fine-tuned on balanced fashion subset with staged unfreezing.</span>
    </div>
    <div class="glass-card">
        <b style="color:#60a5fa;">③ Siamese Network</b><br>
        <span style="color:#94a3b8;font-size:0.9rem;">Metric learning with Contrastive Loss — optimizes pairwise embedding distances for superior visual similarity.</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔍 How to Use</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Navigate to **Image Search** in the sidebar"),
        ("2", "Drag & drop or upload any fashion product image"),
        ("3", "Choose your preferred model & set filters"),
        ("4", "Click **Generate Recommendations** to see results"),
        ("5", "Visit **Model Comparison** to benchmark all three models side-by-side"),
    ]
    for num, step in steps:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.6rem;">'
            f'<span style="background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;font-weight:700;'
            f'border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center;'
            f'font-size:0.78rem;flex-shrink:0;">{num}</span>'
            f'<span style="color:#cbd5e1;font-size:0.92rem;">{step}</span></div>',
            unsafe_allow_html=True
        )

with col2:
    st.markdown('<div class="section-header">🛠️ System Health</div>', unsafe_allow_html=True)
    if val_status["healthy"]:
        st.markdown('<div class="status-ok">● All systems operational</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.success("All directories, embeddings, and model checkpoints verified.")
    else:
        st.markdown('<div class="status-warn">● Some resources missing</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.warning("Some resources need to be built. Check details below.")
        with st.expander("Details Checklist"):
            for label, exists in val_status["dirs"].items():
                icon = "✅" if exists else "❌"
                st.write(f"{icon} Folder: `{label}`")
            for label, exists in val_status["files"].items():
                icon = "✅" if exists else "❌"
                st.write(f"{icon} Database: `{label}`")
            for label, exists in val_status["models"].items():
                icon = "✅" if exists else "❌"
                st.write(f"{icon} Checkpoint: `{label}`")

    st.markdown('<div class="section-header" style="margin-top:1.5rem;">⚡ Tech Stack</div>', unsafe_allow_html=True)
    tags = ["TensorFlow 2.x", "Keras 3", "ResNet50", "FAISS", "Streamlit", "NumPy", "OpenCV", "Scikit-Learn"]
    st.markdown(" ".join(f'<span class="tech-tag">{t}</span>' for t in tags), unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:rgba(148,163,184,0.5);font-size:0.8rem;">'
    '💡 Use the <b style="color:#a78bfa;">Image Search</b> page in the sidebar to begin your visual search.'
    '</div>',
    unsafe_allow_html=True
)

