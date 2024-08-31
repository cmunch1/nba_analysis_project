from ..logging.logging_utils import structured_log
import logging

class NBAScraperError(Exception):
    """Base exception class for NBA scraper errors."""
    def __init__(self, message, log_level=logging.ERROR, **kwargs):
        self.message = message
        self.log_level = log_level
        self.additional_info = kwargs
        self.log()
        super().__init__(self.message)

    def log(self):
        structured_log(logging.getLogger(__name__), self.log_level, self.message, 
                       error_type=self.__class__.__name__, 
                       **self.additional_info)

class ConfigurationError(NBAScraperError):
    """Raised when there's an error in the configuration."""
    def __init__(self, message, log_level=logging.ERROR, **kwargs):
        super().__init__(message, log_level, **kwargs)

class WebDriverError(NBAScraperError):
    """Raised when there's an error with the WebDriver."""
    def __init__(self, message, log_level=logging.ERROR, **kwargs):
        super().__init__(message, log_level, **kwargs)

class ScrapingError(NBAScraperError):
    """Base class for scraping-related errors."""
    def __init__(self, message, log_level=logging.ERROR, **kwargs):
        super().__init__(message, log_level, **kwargs)

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

class DataProcessingError(NBAScraperError):
    """Raised when there's an error processing the scraped data."""
    def __init__(self, message, log_level=logging.ERROR, **kwargs):
        super().__init__(message, log_level, **kwargs)

class DataValidationError(NBAScraperError):
    """Raised when data validation fails."""
    def __init__(self, message, log_level=logging.ERROR, **kwargs):
        super().__init__(message, log_level, **kwargs)

class DataStorageError(NBAScraperError):
    """Raised when there's an error storing or retrieving data."""
    def __init__(self, message, log_level=logging.ERROR, **kwargs):
        super().__init__(message, log_level, **kwargs)

class DynamicContentLoadError(ScrapingError):
    """Raised when dynamic content fails to load within the specified timeout."""
    def __init__(self, message, log_level=logging.ERROR, **kwargs):
        super().__init__(message, log_level, **kwargs)