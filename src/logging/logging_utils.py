import time
import logging
import uuid
import json
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)
error_logger = logging.getLogger('error_logger')

def log_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info("Function performance", extra={
            'function_name': func.__name__,
            'execution_time': f"{end_time - start_time:.2f}",
            'unit': 'seconds'
        })
        return result
    return wrapper

@contextmanager
def log_context(**kwargs):
    original_context = getattr(logging.getLogger(), 'extra', {})
    logger = logging.getLogger()
    try:
        logger.extra = {**original_context, **kwargs}
        yield logger
    finally:
        logger.extra = original_context

def log_error_with_id(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_id = str(uuid.uuid4())
            error_logger.error("An error occurred", extra={
                'error_id': error_id,
                'function_name': func.__name__,
                'error_message': str(e)
            })
            raise
    return wrapper

def get_error_logger():
    return error_logger

def structured_log(logger, level, message, **kwargs):
    """
    Helper function to create structured logs with additional context.
    """
    log_data = {
        'message': message,
        **kwargs
    }
    logger.log(level, json.dumps(log_data))
