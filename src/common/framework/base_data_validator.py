from abc import ABC, abstractmethod
from typing import List
import pandas as pd

from src.common.core.config_management.base_config_manager import BaseConfigManager
from src.common.framework.data_access.base_data_access import BaseDataAccess
from src.common.core.app_logging.base_app_logger import BaseAppLogger
from src.common.core.app_file_handling.base_app_file_handler import BaseAppFileHandler
from src.common.core.error_handling.base_error_handler import BaseErrorHandler

class BaseDataValidator(ABC):
    @abstractmethod
    def __init__(self, config: BaseConfigManager, data_access: BaseDataAccess, app_logger: BaseAppLogger, app_file_handler: BaseAppFileHandler, error_handler: BaseErrorHandler):
        pass

    @abstractmethod
    def validate_scraped_dataframes(self, scraped_dataframes: List[pd.DataFrame], file_names: List[str]) -> bool:
        pass

    @abstractmethod
    def validate_processed_dataframe(self, df: pd.DataFrame, file_name: str) -> bool:
        pass


