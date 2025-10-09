"""
Common test fixtures for nba_app tests.
"""
import pytest
from unittest.mock import Mock

from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.error_handler_factory import ErrorHandlerFactory
from ml_framework.core.error_handling.error_handler import (
    ConfigurationError,
    DataValidationError,
    ScrapingError,
    DataStorageError,
    DataProcessingError,
    FeatureEngineeringError,
    PageLoadError,
    ElementNotFoundError,
    DataExtractionError,
    DynamicContentLoadError
)


@pytest.fixture
def mock_app_logger():
    """Create a mock app logger for testing."""
    logger = Mock(spec=BaseAppLogger)
    logger.structured_log = Mock()
    logger.log_performance = lambda func: func
    logger.log_context = lambda **kwargs: MockContextManager()
    return logger


@pytest.fixture
def mock_error_handler(mock_app_logger):
    """Create a mock error handler factory for testing."""
    handler = Mock(spec=ErrorHandlerFactory)

    def create_error(error_type, message, **kwargs):
        error_map = {
            'configuration': ConfigurationError,
            'data_validation': DataValidationError,
            'scraping': ScrapingError,
            'data_storage': DataStorageError,
            'data_processing': DataProcessingError,
            'feature_engineering': FeatureEngineeringError,
            'page_load': PageLoadError,
            'element_not_found': ElementNotFoundError,
            'data_extraction': DataExtractionError,
            'dynamic_content_load': DynamicContentLoadError
        }
        error_class = error_map.get(error_type, Exception)
        if error_class in error_map.values():
            return error_class(message, mock_app_logger)
        return error_class(message)

    handler.create_error_handler = Mock(side_effect=create_error)
    return handler


class MockContextManager:
    """Mock context manager for logging context."""
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
