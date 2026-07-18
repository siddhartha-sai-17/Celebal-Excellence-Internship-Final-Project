"""
Module Description: Execution Timer Utility
Purpose: Context manager and decorator for measuring execution time and logging performance.
Author: Technical Lead
Version: 1.0.0
Dependencies: time, functools, utils.logger
"""

import time
import functools
from typing import Any, Callable, TypeVar, cast
from utils.logger import performance_logger

F = TypeVar('F', bound=Callable[..., Any])


class Timer:
    """
    A context manager and decorator to log execution times of code blocks and functions.
    """

    def __init__(self, label: str) -> None:
        """
        Initializes the timer with a descriptive label.

        Args:
            label: Descriptive label for the code block or operation.
        """
        self.label: str = label
        self.start_time: float = 0.0
        self.elapsed_time: float = 0.0

    def __enter__(self) -> "Timer":
        """Starts the timer."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Stops the timer and logs the elapsed time.
        """
        self.elapsed_time = time.perf_counter() - self.start_time
        performance_logger.info(f"[Timer] {self.label} completed in {self.elapsed_time:.6f} seconds")


def time_function(label: str = "") -> Callable[[F], F]:
    """
    Decorator to measure and log a function's execution time.

    Args:
        label: Custom label. If empty, the function name will be used.

    Returns:
        Decorated function.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_label = label if label else func.__name__
            with Timer(func_label):
                return func(*args, **kwargs)
        return cast(F, wrapper)
    return decorator
