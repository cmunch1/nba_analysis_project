import logging
import functools
import time
import contextlib
from typing import Callable, Any
from .app_logger import AppLogger, LogContext, _log_context
from .base_app_logger import BaseAppLogger

# Global app logger instance for compatibility
_app_logger_instance = None

def get_app_logger():
    """Get or create the global app logger instance"""
    global _app_logger_instance
    if _app_logger_instance is None:
        _app_logger_instance = AppLogger(None)  # No config for compatibility mode
    return _app_logger_instance

def log_performance(func: Callable) -> Callable:
    """
    Decorator for logging function performance - compatibility function

    Args:
        func: Function to decorate

    Returns:
        Wrapped function with performance logging
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger = logging.getLogger(func.__module__)
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"Function {func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {duration:.3f}s: {e}")
            raise
    return wrapper

def structured_log(logger: logging.Logger, level: int, message: str, **kwargs) -> None:
    """
    Log a structured message with additional context - compatibility function

    Args:
        logger: Logger instance to use
        level: Logging level (e.g., logging.INFO)
        message: Log message
        **kwargs: Additional context to include in log
    """
    # Format message with context if provided
    if kwargs:
        context_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        structured_message = f"{message} | {context_str}"
    else:
        structured_message = message

    logger.log(level, structured_message)

@contextlib.contextmanager
def log_context(**kwargs):
    """
    Context manager for adding context to logs - compatibility function

    Args:
        **kwargs: Context key-value pairs to add to logs
    """
    # For compatibility, this is a no-op context manager
    # The actual context will be handled by individual structured_log calls
    yield

__all__ = ['AppLogger', 'BaseAppLogger', 'LogContext', '_log_context',
           'log_performance', 'structured_log', 'log_context', 'get_app_logger']