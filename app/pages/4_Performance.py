"""
Module Description: Streamlit Performance Dashboard Page
Purpose: Renders CPU/RAM usage panels, latency distribution curves, benchmark summaries, and Plotly radars.
Author: Technical Lead
Version: 3.0.0
Dependencies: streamlit, pandas, matplotlib, json, pathlib, config.settings
"""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import streamlit as st
from config import settings
from app.components.performance_panel import render_performance_panel
from app.components.header import render_header
from app.components.footer import render_footer


def load_json(filepath: Path) -> dict:
    if not filepath.exists():
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


MODEL_META = {
    "baseline":          ("🧠 Baseline CNN",      "#a78bfa"),
    "transfer_learning": ("🔬 Transfer Learning",  "#818cf8"),
    "siamese":           ("⚡ Siamese Network",    "#60a5fa"),
}


def main() -> None:
    """Renders the performance dashboard."""
    from app.utils.session_manager import SessionManager
    from app.utils.theme import inject_custom_css
    SessionManager.initialize_session_state()
    inject_custom_css()

    render_header("⚡ Performance & Diagnostics")

    # ── Real-time resource gauges ──────────────────────────────────
    render_performance_panel()

    st.markdown("---")

    benchmark_file = settings.REPORTS_DIRECTORY / "recommendation_benchmark.json"
    benchmarks = load_json(benchmark_file)

    if not benchmarks:
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:2rem;">'
            '<div style="font-size:2rem;margin-bottom:0.6rem;">📭</div>'
            '<div style="color:#94a3b8;font-size:0.88rem;">No benchmark data found.<br>'
            'Run <code style="color:#a78bfa;">python main.py --action benchmark</code> to generate results.</div>'
            '</div>',
            unsafe_allow_html=True
        )
        render_footer()
        return

    tab_overview, tab_latency, tab_radar, tab_table = st.tabs([
        "📊 Overview",
        "📈 Latency Breakdown",
        "🕸️ Radar Comparison",
        "📋 Full Data Table"
    ])

    # ── Tab 1: Overview Tiles ──────────────────────────────────────
    with tab_overview:
        st.markdown('<div class="section-header">🚀 Model Performance Summary</div>', unsafe_allow_html=True)

        cols = st.columns(len(benchmarks))
        for col, (model_key, metrics) in zip(cols, benchmarks.items()):
            label, color = MODEL_META.get(model_key, (model_key, "#a78bfa"))
            total_ms  = metrics.get("average_recommendation_latency_sec", 0.0) * 1000
            inf_ms    = metrics.get("average_inference_latency_sec", 0.0) * 1000
            search_ms = metrics.get("average_similarity_search_latency_sec", 0.0) * 1000
            cold_s    = metrics.get("cold_start_latency_sec", 0.0)
            fps       = metrics.get("inference_fps", 0.0)
            p50_ms    = metrics.get("p50_latency_sec", 0.0) * 1000
            p95_ms    = metrics.get("p95_latency_sec", 0.0) * 1000

            with col:
                st.markdown(
                    f'<div class="glass-card" style="text-align:center;">'
                    f'<div style="font-weight:700;color:{color};font-size:0.95rem;margin-bottom:0.8rem;'
                    f'border-bottom:1px solid rgba(255,255,255,0.07);padding-bottom:0.6rem;">{label}</div>'
                    # Total latency hero number
                    f'<div style="font-size:2rem;font-weight:800;background:linear-gradient(135deg,{color},#60a5fa);'
                    f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">'
                    f'{total_ms:.0f}<span style="font-size:0.9rem"> ms</span></div>'
                    f'<div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.8rem;">Avg Total Latency</div>'
                    # Sub-metrics grid
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;text-align:left;">'
                    f'<div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:0.5rem;">'
                    f'<div style="font-size:0.68rem;color:#94a3b8;">CNN Inference</div>'
                    f'<div style="font-size:0.88rem;font-weight:700;color:#e2e8f0;">{inf_ms:.1f} ms</div></div>'
                    f'<div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:0.5rem;">'
                    f'<div style="font-size:0.68rem;color:#94a3b8;">Search Latency</div>'
                    f'<div style="font-size:0.88rem;font-weight:700;color:#e2e8f0;">{search_ms:.2f} ms</div></div>'
                    f'<div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:0.5rem;">'
                    f'<div style="font-size:0.68rem;color:#94a3b8;">Cold Start</div>'
                    f'<div style="font-size:0.88rem;font-weight:700;color:#e2e8f0;">{cold_s:.2f} s</div></div>'
                    f'<div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:0.5rem;">'
                    f'<div style="font-size:0.68rem;color:#94a3b8;">Throughput</div>'
                    f'<div style="font-size:0.88rem;font-weight:700;color:#e2e8f0;">{fps:.2f} FPS</div></div>'
                    f'</div>'
                    # P50 / P95 pills
                    f'<div style="display:flex;gap:0.4rem;justify-content:center;margin-top:0.8rem;">'
                    f'<span style="font-size:0.7rem;background:rgba(99,102,241,0.15);color:#a78bfa;'
                    f'padding:2px 10px;border-radius:10px;font-weight:600;">P50: {p50_ms:.0f} ms</span>'
                    f'<span style="font-size:0.7rem;background:rgba(248,113,113,0.12);color:#f87171;'
                    f'padding:2px 10px;border-radius:10px;font-weight:600;">P95: {p95_ms:.0f} ms</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # Optimisation highlights
        st.markdown('<div class="section-header">💡 Optimisation Highlights</div>', unsafe_allow_html=True)
        highlights = [
            ("⚡ FAISS Index",        "Sub-millisecond nearest-neighbour lookup using flat inner-product indexing."),
            ("🔢 Vectorised Cosine",  "NumPy matrix operations avoid Python loops for batch similarity computation."),
            ("💾 Asset Caching",      "Model weights and embedding matrices loaded once and kept in RAM across requests."),
            ("🧊 Frozen Backbone",    "ResNet50 backbone frozen during Siamese training — 22× CPU speedup per step."),
        ]
        hcols = st.columns(2)
        for i, (title, desc) in enumerate(highlights):
            with hcols[i % 2]:
                st.markdown(
                    f'<div class="glass-card" style="margin-bottom:0.6rem;">'
                    f'<div style="font-weight:600;color:#a78bfa;margin-bottom:0.3rem;">{title}</div>'
                    f'<div style="font-size:0.82rem;color:#94a3b8;">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # ── Tab 2: Latency Breakdown Charts ───────────────────────────
    with tab_latency:
        st.markdown('<div class="section-header">📈 Latency Component Breakdown</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.83rem;color:#94a3b8;margin-bottom:1rem;">'
            'Stacked breakdown of CNN inference time vs similarity search vs overhead per model.</p>',
            unsafe_allow_html=True
        )

        plt.style.use("dark_background")

        labels = [MODEL_META[k][0] for k in benchmarks]
        colors_list = [MODEL_META[k][1] for k in benchmarks]
        inf_vals    = [benchmarks[k].get("average_inference_latency_sec", 0) * 1000 for k in benchmarks]
        search_vals = [benchmarks[k].get("average_similarity_search_latency_sec", 0) * 1000 for k in benchmarks]
        total_vals  = [benchmarks[k].get("average_recommendation_latency_sec", 0) * 1000 for k in benchmarks]
        overhead    = [max(t - i - s, 0) for t, i, s in zip(total_vals, inf_vals, search_vals)]

        col_stacked, col_p = st.columns(2)

        with col_stacked:
            fig, ax = plt.subplots(figsize=(6, 4))
            x = np.arange(len(labels))
            w = 0.5
            bar1 = ax.bar(x, inf_vals,    w, label="CNN Inference",    color="#818cf8", alpha=0.9)
            bar2 = ax.bar(x, search_vals, w, bottom=inf_vals,           label="Search",         color="#34d399", alpha=0.9)
            bar3 = ax.bar(x, overhead,    w, bottom=[i+s for i,s in zip(inf_vals, search_vals)], label="Overhead", color="#fbbf24", alpha=0.7)

            ax.set_xticks(x)
            ax.set_xticklabels([l.split(" ", 1)[1] if " " in l else l for l in labels], fontsize=8, color="#94a3b8")
            ax.set_ylabel("Latency (ms)", fontsize=8, color="#94a3b8")
            ax.set_title("Stacked Latency Components", fontsize=10, color="#e2e8f0", pad=10)
            ax.legend(frameon=True, facecolor=(0, 0, 0, 0.5), edgecolor=(1, 1, 1, 0.1), fontsize=8)
            ax.grid(axis="y", color="#ffffff", alpha=0.05, linestyle="--")
            ax.tick_params(colors="#94a3b8")
            fig.patch.set_facecolor("none")
            ax.set_facecolor("none")
            st.pyplot(fig)
            plt.close(fig)

        with col_p:
            # P50 vs P95 grouped bar
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            p50_vals = [benchmarks[k].get("p50_latency_sec", 0) * 1000 for k in benchmarks]
            p95_vals = [benchmarks[k].get("p95_latency_sec", 0) * 1000 for k in benchmarks]
            x2 = np.arange(len(labels))
            ax2.bar(x2 - 0.2, p50_vals, 0.35, label="P50 Latency", color="#6366f1", alpha=0.9)
            ax2.bar(x2 + 0.2, p95_vals, 0.35, label="P95 Latency", color="#f87171", alpha=0.9)
            ax2.set_xticks(x2)
            ax2.set_xticklabels([l.split(" ", 1)[1] if " " in l else l for l in labels], fontsize=8, color="#94a3b8")
            ax2.set_ylabel("Latency (ms)", fontsize=8, color="#94a3b8")
            ax2.set_title("P50 vs P95 Latency Percentiles", fontsize=10, color="#e2e8f0", pad=10)
            ax2.legend(frameon=True, facecolor=(0, 0, 0, 0.5), edgecolor=(1, 1, 1, 0.1), fontsize=8)
            ax2.grid(axis="y", color="#ffffff", alpha=0.05, linestyle="--")
            ax2.tick_params(colors="#94a3b8")
            fig2.patch.set_facecolor("none")
            ax2.set_facecolor("none")
            st.pyplot(fig2)
            plt.close(fig2)

        # Throughput (FPS) bar
        st.markdown('<div class="section-header" style="font-size:1rem;">🎯 Inference Throughput (FPS)</div>', unsafe_allow_html=True)
        fps_vals = {MODEL_META[k][0]: benchmarks[k].get("inference_fps", 0.0) for k in benchmarks}
        fps_df = pd.DataFrame.from_dict(fps_vals, orient="index", columns=["FPS"])
        st.bar_chart(fps_df)

    # ── Tab 3: Radar Comparison ────────────────────────────────────
    with tab_radar:
        st.markdown('<div class="section-header">🕸️ Multi-Metric Radar Chart</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.83rem;color:#94a3b8;margin-bottom:1rem;">'
            'Normalised radar view comparing models across speed, efficiency, and accuracy proxy metrics.</p>',
            unsafe_allow_html=True
        )

        # Load comparison metrics for accuracy
        comp_file = settings.REPORTS_DIRECTORY / "comparison_report.json"
        comp_data = load_json(comp_file)

        # Build radar categories: FPS (speed), 1/TotalLatency (responsiveness), precision@10, ndcg@10, hit_rate@10
        categories = ["Throughput\n(FPS)", "Responsiveness\n(inv-latency)", "Precision\n@10", "NDCG\n@10", "Hit Rate\n@10"]
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig3, ax3 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        fig3.patch.set_facecolor("none")
        ax3.set_facecolor("none")

        max_fps = max((benchmarks[k].get("inference_fps", 1) for k in benchmarks), default=1)
        max_resp = max((1 / max(benchmarks[k].get("average_recommendation_latency_sec", 1), 1e-9) for k in benchmarks), default=1)

        for model_key in benchmarks:
            label, color = MODEL_META.get(model_key, (model_key, "#a78bfa"))
            fps_norm  = benchmarks[model_key].get("inference_fps", 0) / max(max_fps, 1e-9)
            resp_norm = (1 / max(benchmarks[model_key].get("average_recommendation_latency_sec", 1), 1e-9)) / max(max_resp, 1e-9)
            prec_10   = comp_data.get(model_key, {}).get("precision_at_10", 0)
            ndcg_10   = comp_data.get(model_key, {}).get("ndcg_at_10", 0)
            hit_10    = comp_data.get(model_key, {}).get("hit_rate_at_10", 0)

            values = [fps_norm, resp_norm, prec_10, ndcg_10, hit_10]
            values += values[:1]

            ax3.plot(angles, values, "o-", linewidth=2, color=color, label=label)
            ax3.fill(angles, values, alpha=0.12, color=color)

        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(categories, color="#94a3b8", fontsize=8)
        ax3.yaxis.set_tick_params(labelcolor="#94a3b8", labelsize=7)
        ax3.spines["polar"].set_color((1, 1, 1, 0.08))
        ax3.grid(color="#ffffff", alpha=0.06)
        ax3.legend(
            loc="upper right", bbox_to_anchor=(1.3, 1.1),
            frameon=True, facecolor=(0, 0, 0, 0.6),
            edgecolor=(1, 1, 1, 0.1), fontsize=8
        )
        ax3.set_title("Normalised Model Comparison Radar", fontsize=10, color="#e2e8f0", pad=20)
        col_radar, col_radar_info = st.columns([2, 1])
        with col_radar:
            st.pyplot(fig3)
            plt.close(fig3)
        with col_radar_info:
            st.markdown(
                '<div class="glass-card">'
                '<div style="font-size:0.82rem;font-weight:600;color:#a78bfa;margin-bottom:0.8rem;">Radar Axis Guide</div>'
                '<div style="font-size:0.78rem;color:#94a3b8;line-height:1.8;">'
                '• <b style="color:#e2e8f0;">Throughput (FPS)</b>: Images processed per second<br>'
                '• <b style="color:#e2e8f0;">Responsiveness</b>: Inverse of total recommendation latency<br>'
                '• <b style="color:#e2e8f0;">Precision@10</b>: Fraction of top-10 that are relevant<br>'
                '• <b style="color:#e2e8f0;">NDCG@10</b>: Rank-weighted relevance quality<br>'
                '• <b style="color:#e2e8f0;">Hit Rate@10</b>: Probability of ≥1 relevant in top-10<br>'
                '<br>All axes normalised 0–1.</div>'
                '</div>',
                unsafe_allow_html=True
            )

    # ── Tab 4: Full Data Table ─────────────────────────────────────
    with tab_table:
        st.markdown('<div class="section-header">📋 Detailed Benchmark Table</div>', unsafe_allow_html=True)
        rows = []
        for model, metrics in benchmarks.items():
            label, _ = MODEL_META.get(model, (model, "#a78bfa"))
            rows.append({
                "Model":               label,
                "Cold Start (s)":      f"{metrics.get('cold_start_latency_sec', 0.0):.3f}",
                "CNN Inference (ms)":  f"{metrics.get('average_inference_latency_sec', 0.0)*1000:.1f}",
                "Search (ms)":         f"{metrics.get('average_similarity_search_latency_sec', 0.0)*1000:.3f}",
                "Total Avg (ms)":      f"{metrics.get('average_recommendation_latency_sec', 0.0)*1000:.1f}",
                "P50 (ms)":            f"{metrics.get('p50_latency_sec', 0.0)*1000:.1f}",
                "P95 (ms)":            f"{metrics.get('p95_latency_sec', 0.0)*1000:.1f}",
                "FPS":                 f"{metrics.get('inference_fps', 0.0):.2f}",
            })
        st.table(pd.DataFrame(rows))

    render_footer()


if __name__ == "__main__":
    import pandas as pd
    main()
