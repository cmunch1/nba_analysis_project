import time
import logging
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger(__name__)

def log_performance(func):
    """Decorator to log the performance of a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

@contextmanager
def log_context(**kwargs):
    """Add context to logs within this block"""
    original_context = getattr(logging.getLogger(), 'extra', {})
    logger = logging.getLogger()
    try:
        logger.extra = {**original_context, **kwargs}
        yield logger
    finally:
        logger.extra = original_context

