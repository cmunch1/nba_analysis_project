# Factory for creating loggers
from src.common.app_logging.base_app_logger import BaseAppLogger

from src.common.app_logging.app_logger import AppLogger

class AppLoggerFactory:
    @staticmethod
    def create_app_logger() -> BaseAppLogger:
        """
        Create and return a configured app logger instance
        
        Args:
            config: Configuration object
            
        Returns:
            Configured app logger instance
        """
        return AppLogger()