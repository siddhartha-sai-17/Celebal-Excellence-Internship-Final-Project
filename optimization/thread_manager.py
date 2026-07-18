"""
Module Description: Background Thread Manager
Purpose: Executes long-running tasks (subsetting, training, benchmarks) asynchronously in background threads.
Author: Technical Lead
Version: 1.0.0
Dependencies: concurrent.futures, typing, utils.logger
"""

from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Any, Dict, Optional
from utils.logger import app_logger


class ThreadManager:
    """
    Manages background worker threads to run computational tasks without blocking the presentation layer.
    """
    _instance: Optional["ThreadManager"] = None
    _executor: Optional[ThreadPoolExecutor] = None
    _active_futures: Dict[str, Future] = {}

    def __new__(cls, *args, **kwargs) -> "ThreadManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initializes the thread pool executor if not already done."""
        if self._executor is None:
            # Re-use thread pool executor with maximum 4 workers
            self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="BgWorker")
            app_logger.info("Background ThreadPoolExecutor initialized with max_workers=4.")

    def submit_task(self, task_id: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
        """
        Submits a callable task to run asynchronously in the background.

        Args:
            task_id: Unique string label for this task.
            fn: Function to execute.
            args: Positional arguments for fn.
            kwargs: Keyword arguments for fn.

        Returns:
            True if task was successfully queued, False if a task with task_id is already active.
        """
        if task_id in self._active_futures and not self._active_futures[task_id].done():
            app_logger.warning(f"Task with ID '{task_id}' is already running in background.")
            return False

        app_logger.info(f"Submitting background task: {task_id}")
        future = self._executor.submit(fn, *args, **kwargs)
        self._active_futures[task_id] = future
        return True

    def is_task_running(self, task_id: str) -> bool:
        """Checks if a task is currently executing."""
        if task_id not in self._active_futures:
            return False
        return not self._active_futures[task_id].done()

    def get_task_result(self, task_id: str) -> Optional[Any]:
        """
        Retrieves the result of a completed background task.

        Args:
            task_id: ID of target task.

        Returns:
            Result of the task, or None if task is not found, not completed, or failed.
        """
        if task_id not in self._active_futures:
            return None
        
        future = self._active_futures[task_id]
        if not future.done():
            return None

        try:
            return future.result()
        except Exception as e:
            app_logger.error(f"Task '{task_id}' failed with exception: {e}")
            return None

    def shutdown(self, wait: bool = True) -> None:
        """Shuts down the thread pool."""
        if self._executor:
            self._executor.shutdown(wait=wait)
            app_logger.info("ThreadPoolExecutor shut down successfully.")
