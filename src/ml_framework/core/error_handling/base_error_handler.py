from abc import ABC, abstractmethod
import logging
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger

class BaseErrorHandler(ABC, Exception):
    def __init__(self, message: str, app_logger: BaseAppLogger, log_level: int = logging.ERROR, **kwargs):
        self.app_logger = app_logger
        self.message = message
        self.log_level = log_level
        self.additional_info = kwargs
        self.log()
        # Properly call Exception's init
        Exception.__init__(self, message)

    @abstractmethod
    def log(self) -> None:
        """
        Log the error with the specified level and additional information.
        Must be implemented by concrete classes.
        """
        pass