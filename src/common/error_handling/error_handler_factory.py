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
    
    def __init__(self, app_logger: BaseAppLogger):
        """
        Initialize the error handler factory.
        
        Args:
            logger (BaseAppLogger): Logger instance to be injected into error handlers
        """
        self.app_logger = app_logger
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
                app_logger=self.app_logger,
                log_level=log_level,
                **kwargs
            )
        else:
            return error_class(
                message=message,
                app_logger=self.app_logger,
                **kwargs
            )