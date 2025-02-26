"""
Preprocessor Factory Module

This module provides a factory for creating preprocessor instances with
proper dependency injection aligned with the application architecture.
"""

import logging
from typing import Type

from src.common.config_management.base_config_manager import BaseConfigManager
from src.common.app_logging.base_app_logger import BaseAppLogger
from src.common.error_handling.base_error_handler import BaseErrorHandler
from src.common.app_file_handling.base_app_file_handler import BaseAppFileHandler
from .base_preprocessor import BasePreprocessor
from .preprocessor import Preprocessor


class PreprocessorFactory:
    """Factory for creating preprocessor instances with proper dependency injection."""
    
    @classmethod
    def create_preprocessor(cls,
                          config: BaseConfigManager,
                          app_logger: BaseAppLogger,
                          app_file_handler: BaseAppFileHandler,
                          error_handler: BaseErrorHandler,
                          preprocessor_type: str = "modular") -> BasePreprocessor:
        """
        Create a preprocessor instance with dependencies.
        
        Args:
            config: Configuration manager
            app_logger: Application logger
            app_file_handler: File handler
            error_handler: Error handler
            preprocessor_type: Type of preprocessor to create
            
        Returns:
            BasePreprocessor: Configured preprocessor instance
            
        Raises:
            ValueError: If preprocessor_type is not supported
        """
        app_logger.structured_log(
            logging.INFO,
            "Creating preprocessor",
            preprocessor_type=preprocessor_type
        )
        
        if preprocessor_type.lower() == "modular":
            return Preprocessor(
                config=config,
                app_logger=app_logger,
                app_file_handler=app_file_handler,
                error_handler=error_handler
            )
        else:
            raise ValueError(f"Unsupported preprocessor type: {preprocessor_type}")

    @classmethod
    def register_preprocessor(cls, name: str, preprocessor_class: Type[BasePreprocessor]) -> None:
        """
        Register a new preprocessor type.
        
        Args:
            name: Name to register the preprocessor under
            preprocessor_class: Preprocessor class to register
        """
        if not issubclass(preprocessor_class, BasePreprocessor):
            raise TypeError(f"Preprocessor class must inherit from BasePreprocessor: {preprocessor_class}")
            
        setattr(cls, f"create_{name}_preprocessor", 
               lambda config, app_logger, app_file_handler, error_handler: 
               preprocessor_class(config, app_logger, app_file_handler, error_handler))