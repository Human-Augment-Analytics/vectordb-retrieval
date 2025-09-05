import time
from functools import wraps
from typing import Callable, Any, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

def time_function(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.

    Args:
        func: Function to time

    Returns:
        Wrapped function that logs execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"{func.__name__} completed in {elapsed_time:.6f} seconds")
        return result
    return wrapper

class Timer:
    """
    Context manager for timing code blocks.
    """
    def __init__(self, name: str = "Operation"):
        """
        Initialize the timer.

        Args:
            name: Name of the operation being timed
        """
        self.name = name
        self.start_time = None
        self.elapsed_time = None

    def __enter__(self) -> 'Timer':
        """
        Start the timer when entering the context.

        Returns:
            Self for accessing the timer in the context
        """
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Stop the timer when exiting the context and log the elapsed time.
        """
        end_time = time.time()
        self.elapsed_time = end_time - self.start_time
        logger.info(f"{self.name} completed in {self.elapsed_time:.6f} seconds")
