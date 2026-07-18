"""
Module Description: System Performance Telemetry Panel
Purpose: Renders resource usage (CPU, RAM, Disk, Active threads) telemetry charts and metrics.
Author: Technical Lead
Version: 1.0.0
Dependencies: streamlit, optimization.resource_monitor
"""

import streamlit as st
from optimization.resource_monitor import ResourceMonitor


def render_performance_panel() -> None:
    """
    Queries the resource monitor and draws CPU/RAM/Disk consumption gauges.
    """
    st.markdown("### 🖥️ Real-time Resource Telemetry")

    # Get metrics
    metrics = ResourceMonitor.get_system_metrics()

    col1, col2, col3 = st.columns(3)

    with col1:
        cpu = metrics.get("cpu_percent", 0.0)
        st.metric(label="CPU Utilization", value=f"{cpu}%")
        st.progress(cpu / 100.0)

    with col2:
        ram = metrics.get("memory_percent", 0.0)
        used_ram = metrics.get("memory_used_gb", 0.0)
        total_ram = metrics.get("memory_total_gb", 0.0)
        st.metric(label="RAM Usage", value=f"{ram}%", delta=f"{used_ram} / {total_ram} GB")
        st.progress(ram / 100.0)

    with col3:
        disk = metrics.get("disk_percent", 0.0)
        free_disk = metrics.get("disk_free_gb", 0.0)
        st.metric(label="Disk Capacity Used", value=f"{disk}%", delta=f"{free_disk} GB Free", delta_color="normal")
        st.progress(disk / 100.0)

    st.markdown("---")
    
    # Extra system details
    col_a, col_b = st.columns(2)
    with col_a:
        st.info(f"**Active Background Threads**: {metrics.get('active_threads', 1)}")
    with col_b:
        gpu_active = metrics.get("gpu_available", False)
        status_text = f"🟢 Available ({metrics.get('gpu_name')})" if gpu_active else "🔴 Unavailable (Using CPU Fallback)"
        st.info(f"**Hardware GPU Acceleration**: {status_text}")
