from abc import ABC, abstractmethod
from typing import Callable
import logging
import contextlib

from src.common.config_management.base_config_manager import BaseConfigManager

class BaseAppLogger(ABC):
    """Abstract base class defining logging interface"""
    
    @abstractmethod
    def __init__(self, config: BaseConfigManager): 
        """Initialize logger with configuration"""
        pass
        
    @abstractmethod
    def setup(self, log_file: str) -> logging.Logger:
        """Setup logging configuration"""
        pass
        
    @abstractmethod
    def structured_log(self, level: int, message: str, **kwargs) -> None:
        """Log a structured message with additional context"""
        pass
        
    @abstractmethod
    def log_performance(self, func: Callable) -> Callable:
        """Decorator for logging function performance"""
        pass
        
    @abstractmethod
    def log_context(self, **kwargs) -> contextlib.AbstractContextManager:
        """Context manager for adding context to logs"""
        pass

