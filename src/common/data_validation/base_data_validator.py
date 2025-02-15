from abc import ABC, abstractmethod
from typing import List
import pandas as pd

from ..config_management.base_config_manager import BaseConfigManager
from ..data_access.base_data_access import BaseDataAccess
from ..app_logging.base_app_logger import BaseAppLogger
from ..app_file_handling.base_app_file_handler import BaseAppFileHandler
from ..error_handling.base_error_handler import BaseErrorHandler

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


