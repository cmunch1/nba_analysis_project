from typing import Type, Optional
from src.common.app_logging.base_app_logger import BaseAppLogger
from src.common.error_handling.error_handler import (
    ConfigurationError,
    WebDriverError,
    ScrapingError,
    PageLoadError,
    ElementNotFoundError,
    DataExtractionError,
    DataProcessingError,
    DataValidationError,
    DataStorageError,
    DynamicContentLoadError,
    FeatureEngineeringError,
    FeatureSelectionError,
    ModelTestingError,
    ChartCreationError,
    PreprocessingError,
    OptimizationError,
    ExperimentLoggerError
)

class ErrorHandlerFactory:
    """Factory for creating error handler instances with proper logging configuration."""
    
    def __init__(self, logger: BaseAppLogger):
        """
        Initialize the error handler factory.
        
        Args:
            logger (BaseAppLogger): Logger instance to be injected into error handlers
        """
        self.logger = logger
        self._error_classes = {
            # Core errors
            'configuration': ConfigurationError,
            'webdriver': WebDriverError,
            
            # Scraping related errors
            'scraping': ScrapingError,
            'page_load': PageLoadError,
            'element_not_found': ElementNotFoundError,
            'data_extraction': DataExtractionError,
            'dynamic_content_load': DynamicContentLoadError,
            
            # Data related errors
            'data_processing': DataProcessingError,
            'data_validation': DataValidationError,
            'data_storage': DataStorageError,
            
            # ML/Analytics related errors
            'feature_engineering': FeatureEngineeringError,
            'feature_selection': FeatureSelectionError,
            'model_testing': ModelTestingError,
            'preprocessing': PreprocessingError,
            'optimization': OptimizationError,
            
            # Visualization errors
            'chart_creation': ChartCreationError,
            
            # Logging errors
            'experiment_logger': ExperimentLoggerError,
        }
    
    def create_error_handler(
        self,
        error_type: str,
        message: str,
        log_level: Optional[int] = None,
        **kwargs
    ) -> Type[Exception]:
        """
        Create an error handler instance of the specified type.
        
        Args:
            error_type (str): Type of error handler to create
            message (str): Error message
            log_level (Optional[int]): Logging level. If None, uses default for error type
            **kwargs: Additional information to be included in the error log
        
        Returns:
            Type[Exception]: Instantiated error handler
            
        Raises:
            ValueError: If error_type is not recognized
        """
        if error_type not in self._error_classes:
            valid_types = ", ".join(sorted(self._error_classes.keys()))
            raise ValueError(
                f"Unknown error type: {error_type}. "
                f"Valid types are: {valid_types}"
            )
        
        error_class = self._error_classes[error_type]
        
        # Create error handler with injected logger
        if log_level is not None:
            return error_class(
                message=message,
                logger=self.logger,
                log_level=log_level,
                **kwargs
            )
        else:
            return error_class(
                message=message,
                logger=self.logger,
                **kwargs
            )
    
    def register_error_class(
        self,
        error_type: str,
        error_class: Type[Exception]
    ) -> None:
        """
        Register a new error class with the factory.
        
        Args:
            error_type (str): Name to register the error class under
            error_class (Type[Exception]): Error class to register
            
        Raises:
            ValueError: If error_type is already registered
        """
        if error_type in self._error_classes:
            raise ValueError(f"Error type {error_type} is already registered")
        self._error_classes[error_type] = error_class

    def get_registered_error_types(self) -> list[str]:
        """
        Get a list of all registered error types.
        
        Returns:
            list[str]: Sorted list of registered error type names
        """
        return sorted(self._error_classes.keys())