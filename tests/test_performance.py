"""
Module Description: Performance Diagnostic Unit Tests
Purpose: Assures memory optimization, CPU checks, and diagnostic logs perform correctly.
Author: Technical Lead
Version: 1.0.0
Dependencies: pytest, optimization.memory_optimizer, optimization.resource_monitor, optimization.system_health
"""

import pytest
from optimization.memory_optimizer import MemoryOptimizer
from optimization.resource_monitor import ResourceMonitor
from optimization.system_health import SystemHealth


def test_memory_optimizer() -> None:
    """Ensures garbage collection and tf backend clear calls run without crash."""
    try:
        freed = MemoryOptimizer.force_garbage_collection()
        assert isinstance(freed, float)
        assert freed >= 0.0
        MemoryOptimizer.log_system_memory_report()
    except Exception as e:
        pytest.fail(f"Memory cleanup execution failed: {e}")


def test_resource_monitor() -> None:
    """Ensures resources statistics dictionary can be compiled."""
    metrics = ResourceMonitor.get_system_metrics()
    assert isinstance(metrics, dict)
    assert "cpu_percent" in metrics
    assert "memory_percent" in metrics
    assert "active_threads" in metrics
    assert metrics["active_threads"] >= 1


def test_system_health() -> None:
    """Ensures system telemetry diagnostic runs without exceptions."""
    health = SystemHealth.get_health_status()
    assert isinstance(health, dict)
    assert "status" in health
    assert health["status"] in ("Healthy", "Warning")
    assert "cpu_warning" in health
    assert "memory_warning" in health
