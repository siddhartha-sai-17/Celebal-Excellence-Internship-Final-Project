"""
Module Description: Streamlit UI Theme Styles
Purpose: Centralizes premium dark glassmorphism CSS templates for styling, animations, card styling, and dark mode tweaks.
Author: Technical Lead
Version: 2.0.0
Dependencies: streamlit
"""

import streamlit as st


def inject_custom_css() -> None:
    """
    Applies premium dark glassmorphism CSS design system into Streamlit page layout.
    """
    css_styles = """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        /* ── Global Font ── */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        /* ── Hide Streamlit chrome ── */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* ── Page Background ── */
        .stApp {
            background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #0a1628 70%, #0e1a2e 100%);
            min-height: 100vh;
        }

        /* ── Container padding ── */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        /* ── Hero Banner ── */
        .hero-banner {
            background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(168,85,247,0.10) 50%, rgba(59,130,246,0.12) 100%);
            border: 1px solid rgba(99,102,241,0.3);
            border-radius: 20px;
            padding: 2.5rem 2rem;
            margin-bottom: 2rem;
            backdrop-filter: blur(12px);
            position: relative;
            overflow: hidden;
        }
        .hero-banner::before {
            content: '';
            position: absolute;
            top: -50%; left: -50%;
            width: 200%; height: 200%;
            background: radial-gradient(circle at 30% 50%, rgba(99,102,241,0.08) 0%, transparent 60%),
                        radial-gradient(circle at 70% 20%, rgba(168,85,247,0.06) 0%, transparent 50%);
            animation: aurora 8s ease infinite alternate;
        }
        @keyframes aurora {
            0%   { transform: translate(0,0) rotate(0deg); }
            100% { transform: translate(3%,2%) rotate(3deg); }
        }
        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #a78bfa 0%, #818cf8 40%, #60a5fa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0 0 0.5rem 0;
            line-height: 1.2;
            position: relative;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: rgba(148,163,184,0.9);
            font-weight: 400;
            margin: 0;
            position: relative;
        }

        /* ── Section Headers ── */
        .section-header {
            font-size: 1.3rem;
            font-weight: 700;
            color: #e2e8f0;
            border-left: 4px solid #6366f1;
            padding-left: 0.85rem;
            margin: 1.8rem 0 1rem 0;
        }

        /* ── Glassmorphism Cards ── */
        .glass-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
            margin-bottom: 1rem;
        }
        .glass-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.4), 0 0 0 1px rgba(99,102,241,0.3);
            border-color: rgba(99,102,241,0.35);
        }

        /* ── Product Cards ── */
        .product-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1rem;
            transition: transform 0.25s cubic-bezier(0.34,1.56,0.64,1), box-shadow 0.25s ease, border-color 0.25s ease;
            margin-bottom: 1.2rem;
            height: 100%;
        }
        .product-card:hover {
            transform: translateY(-6px) scale(1.01);
            box-shadow: 0 24px 48px rgba(0,0,0,0.5), 0 0 0 1px rgba(99,102,241,0.4);
            border-color: rgba(99,102,241,0.5);
        }
        .product-card img {
            border-radius: 10px;
            object-fit: cover;
            width: 100%;
            height: 200px;
            margin-bottom: 0.8rem;
        }

        /* ── Similarity Badge ── */
        .similarity-badge {
            background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(168,85,247,0.20));
            color: #a78bfa;
            font-weight: 700;
            font-size: 0.78rem;
            padding: 4px 10px;
            border-radius: 20px;
            border: 1px solid rgba(167,139,250,0.3);
            display: inline-block;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        /* ── Metric Tiles ── */
        .metric-tile {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 1.2rem 1.5rem;
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-tile:hover { transform: translateY(-3px); box-shadow: 0 12px 28px rgba(0,0,0,0.35); }
        .metric-tile .value {
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #a78bfa, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            display: block;
            line-height: 1.1;
        }
        .metric-tile .label {
            font-size: 0.78rem;
            color: rgba(148,163,184,0.8);
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-top: 0.4rem;
            display: block;
        }

        /* ── Status Pills ── */
        .status-ok  { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3);  border-radius: 20px; padding: 3px 12px; font-size: 0.78rem; font-weight: 600; display: inline-block; }
        .status-warn{ background: rgba(245,158,11,0.15);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.3);  border-radius: 20px; padding: 3px 12px; font-size: 0.78rem; font-weight: 600; display: inline-block; }
        .status-err { background: rgba(239,68,68,0.15);   color: #f87171; border: 1px solid rgba(248,113,113,0.3); border-radius: 20px; padding: 3px 12px; font-size: 0.78rem; font-weight: 600; display: inline-block; }

        /* ── Metric Block ── */
        .metric-block {
            border-left: 3px solid #6366f1;
            padding: 0.8rem 1rem;
            background: rgba(99,102,241,0.06);
            border-radius: 0 10px 10px 0;
            margin-bottom: 1rem;
        }

        /* ── Shimmer Skeleton ── */
        .skeleton-box {
            display: inline-block;
            height: 200px; width: 100%;
            background: linear-gradient(90deg, rgba(255,255,255,0.04) 0px, rgba(255,255,255,0.10) 40px, rgba(255,255,255,0.04) 80px);
            background-size: 600px;
            animation: shimmer 1.6s infinite linear;
            border-radius: 12px;
        }
        @keyframes shimmer {
            0%   { background-position: -300px 0; }
            100% { background-position: 600px 0; }
        }

        /* ── Tech Tag Pills ── */
        .tech-tag {
            background: rgba(99,102,241,0.12);
            border: 1px solid rgba(99,102,241,0.25);
            color: #a5b4fc;
            border-radius: 6px;
            padding: 3px 10px;
            font-size: 0.78rem;
            font-weight: 500;
            display: inline-block;
            margin: 3px;
        }

        /* ── Legacy header support ── */
        .custom-header {
            font-size: 2rem; font-weight: 800;
            background: linear-gradient(135deg, #a78bfa 0%, #818cf8 50%, #60a5fa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.3rem;
        }
        .custom-subheader { font-size: 1rem; color: rgba(148,163,184,0.8); margin-bottom: 1.5rem; }

        /* ── Native Streamlit overrides ── */
        .stButton > button {
            background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
            color: white !important; border: none !important;
            border-radius: 10px !important; font-weight: 600 !important;
            padding: 0.6rem 1.6rem !important;
            box-shadow: 0 4px 15px rgba(99,102,241,0.35) !important;
            transition: opacity 0.2s, transform 0.15s !important;
        }
        .stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important; }
        .stButton > button[kind="secondary"] { background: rgba(255,255,255,0.06) !important; border: 1px solid rgba(255,255,255,0.12) !important; box-shadow: none !important; }

        [data-testid="metric-container"] {
            background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px; padding: 1rem 1.2rem;
        }
        [data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 800 !important; color: #a78bfa !important; }
        [data-testid="metric-container"] [data-testid="stMetricLabel"] { font-size: 0.78rem !important; color: rgba(148,163,184,0.8) !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.05em; }

        [data-testid="stSidebar"] { background: rgba(10,14,26,0.97) !important; border-right: 1px solid rgba(255,255,255,0.06) !important; }

        [data-testid="stTable"] th { background: rgba(99,102,241,0.15) !important; color: #a78bfa !important; font-weight: 600 !important; font-size: 0.82rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
        [data-testid="stTable"] td { color: #cbd5e1 !important; font-size: 0.88rem !important; }
        [data-testid="stTable"] tr:hover td { background: rgba(99,102,241,0.06) !important; }

        [data-testid="stExpander"] { background: rgba(255,255,255,0.03) !important; border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 12px !important; }
        [data-testid="stFileUploader"] { background: rgba(255,255,255,0.03) !important; border: 2px dashed rgba(99,102,241,0.35) !important; border-radius: 16px !important; }
        [data-testid="stFileUploader"]:hover { border-color: rgba(99,102,241,0.7) !important; }
        .stSpinner > div > div { border-top-color: #6366f1 !important; }
        hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.5rem 0 !important; }

        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
        ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.7); }

        .fade-in { animation: fadeIn 0.5s ease forwards; }
        @keyframes fadeIn { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
    </style>
    """
    st.markdown(css_styles, unsafe_allow_html=True)

