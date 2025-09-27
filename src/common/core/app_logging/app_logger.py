import logging
import time
import functools
import contextlib
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass
from contextvars import ContextVar
from src.common.core.config_management.config_manager import BaseConfigManager
from src.common.core.app_logging.base_app_logger import BaseAppLogger


# Context variable to store logging context
_log_context: ContextVar[Dict[str, Any]] = ContextVar('log_context', default={})

@dataclass
class LogContext:
    """Class to store logging context data"""
    app_version: str
    environment: str
    additional_context: Dict[str, Any] = None


class AppLogger(BaseAppLogger):
    """Concrete implementation of BaseAppLogger"""
    
    def __init__(self, config: BaseConfigManager):
        self.logger = None
        self.config = config
        
    def setup(self, log_file: str) -> logging.Logger:
        """
        Setup logging configuration with file and console handlers
        
        Args:
            log_file: Path to log file
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(__name__)
        
        # Use config log level if available, otherwise default to INFO
        log_level = getattr(self.config, 'log_level', logging.INFO) if self.config else logging.INFO
        logger.setLevel(log_level)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # File handler
        log_file_handler = logging.FileHandler(log_file)
        log_file_handler.setFormatter(file_formatter)
        logger.addHandler(log_file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        self.logger = logger
        return logger
        
    def structured_log(self, level: int, message: str, **kwargs) -> None:
        """
        Log a structured message with additional context
        
        Args:
            level: Logging level (e.g., logging.INFO)
            message: Log message
            **kwargs: Additional context to include in log
        """
        if self.logger is None:
            raise RuntimeError("Logger not initialized. Call setup() first.")
            
        # Get current context
        context = _log_context.get()
        
        # Combine context with kwargs
        log_data = {**context, **kwargs}
        
        # Format message with context
        structured_message = f"{message} | Context: {log_data}"
        
        self.logger.log(level, structured_message)
        
    def log_performance(self, func: Callable) -> Callable:
        """
        Decorator for logging function performance
        
        Args:
            func: Function to decorate
            
        Returns:
            Wrapped function with performance logging
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                self.structured_log(
                    logging.INFO,
                    f"Function {func.__name__} completed",
                    duration_seconds=duration,
                    status="success"
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                self.structured_log(
                    logging.ERROR,
                    f"Function {func.__name__} failed",
                    duration_seconds=duration,
                    status="error",
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                raise
        return wrapper
        
    @contextlib.contextmanager
    def log_context(self, **kwargs):
        """
        Context manager for adding context to logs
        
        Args:
            **kwargs: Context key-value pairs to add to logs
        """
        token = None
        try:
            # Get existing context and update with new values
            current_context = _log_context.get()
            new_context = {**current_context, **kwargs}
            
            # Set new context
            token = _log_context.set(new_context)
            yield
        finally:
            # Restore previous context
            if token is not None:
                _log_context.reset(token)

