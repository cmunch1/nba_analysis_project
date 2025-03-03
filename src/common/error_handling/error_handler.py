from src.common.app_logging.base_app_logger import BaseAppLogger
from .base_error_handler import BaseErrorHandler
import logging

class ErrorHandler(BaseErrorHandler):   
    exit_code = 1  # Default exit code
    
    def __init__(self, message: str, app_logger: BaseAppLogger, log_level=logging.ERROR, **kwargs):
        # Just call the base class init properly
        super().__init__(message, app_logger, log_level, **kwargs)

    def log(self) -> None:
        """Implementation of abstract log method"""
        self.app_logger.structured_log(
            self.log_level, 
            self.message,
            error_type=self.__class__.__name__, 
            **self.additional_info
        )

class ConfigurationError(ErrorHandler):
    """Raised when there's an error in the configuration."""
    exit_code = 2

class WebDriverError(ErrorHandler):
    """Raised when there's an error with the WebDriver."""
    exit_code = 3

class ScrapingError(ErrorHandler):
    """Base class for scraping-related errors."""
    exit_code = 4

class PageLoadError(ScrapingError):
    """Raised when a page fails to load."""
    def __init__(self, message, log_level=logging.ERROR, **kwargs):
        super().__init__(message, log_level, **kwargs)

class ElementNotFoundError(ScrapingError):
    """Raised when an expected element is not found on the page."""
    def __init__(self, message, log_level=logging.ERROR, **kwargs):
        super().__init__(message, log_level, **kwargs)

class DataExtractionError(ScrapingError):
    """Raised when there's an error extracting data from the page."""
    def __init__(self, message, log_level=logging.ERROR, **kwargs):
        super().__init__(message, log_level, **kwargs)

class DataProcessingError(ScrapingError):
    """Raised when there's an error processing the scraped data."""
    exit_code = 5

class DataValidationError(ErrorHandler):
    """Raised when data validation fails."""
    def __init__(self, message, log_level=logging.ERROR, **kwargs):
        super().__init__(message, log_level, **kwargs)

class DataStorageError(ErrorHandler):
    """Raised when there's an error storing or retrieving data."""
    def __init__(self, message, log_level=logging.ERROR, **kwargs):
        super().__init__(message, log_level, **kwargs)

class DynamicContentLoadError(ScrapingError):
    """Raised when dynamic content fails to load within the specified timeout."""
    def __init__(self, message, log_level=logging.ERROR, **kwargs):
        super().__init__(message, log_level, **kwargs)

class FeatureEngineeringError(ErrorHandler):
    """Raised when there's an error in the feature engineering process."""
    def __init__(self, message, log_level=logging.ERROR, **kwargs):
        super().__init__(message, log_level, **kwargs)

class FeatureSelectionError(ErrorHandler):
    """Raised when there's an error in the feature selection process."""
    def __init__(self, message, log_level=logging.ERROR, **kwargs):
        super().__init__(message, log_level, **kwargs)

class ModelTestingError(ErrorHandler):
    """Raised when there's an error in the model testing process."""
    def __init__(self, message, log_level=logging.ERROR, **kwargs):
        super().__init__(message, log_level, **kwargs)

class ChartCreationError(ErrorHandler):
    """Raised when there's an error in the chart creation process."""
    def __init__(self, message, log_level=logging.ERROR, **kwargs):
        super().__init__(message, log_level, **kwargs)

class PreprocessingError(ErrorHandler):
    """Raised when there's an error in the preprocessing process."""
    def __init__(self, message, log_level=logging.ERROR, **kwargs):
        super().__init__(message, log_level, **kwargs)
 
class OptimizationError(ErrorHandler):
    """Raised when there's an error in the optimization process."""
    def __init__(self, message, log_level=logging.ERROR, **kwargs):
        super().__init__(message, log_level, **kwargs)

class ExperimentLoggerError(ErrorHandler):
    """Raised when there's an error in the experiment logger process."""
    def __init__(self, message, log_level=logging.ERROR, **kwargs):
        super().__init__(message, log_level, **kwargs)
