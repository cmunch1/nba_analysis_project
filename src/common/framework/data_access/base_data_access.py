"""
abstract_data_access.py

This file contains abstract base classes for data access operations.
These classes define the interface for data access operations, which can be implemented
by concrete classes for different data sources (e.g., CSV files, databases, APIs).
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Tuple, Dict

from src.common.core.config_management.base_config_manager import BaseConfigManager
from src.common.core.app_logging.base_app_logger import BaseAppLogger
from src.common.core.app_file_handling.base_app_file_handler import BaseAppFileHandler
from src.common.core.error_handling.base_error_handler import BaseErrorHandler



class BaseDataAccess(ABC):

    @abstractmethod
    def __init__(self, config: BaseConfigManager, app_logger: BaseAppLogger, 
                 app_file_handler: BaseAppFileHandler, error_handler: BaseErrorHandler):
        pass

    @abstractmethod
    def save_dataframes(self, dataframes: List[pd.DataFrame], file_names: List[str], cumulative: bool = False) -> None:
        """
        Save the list of dataframes to separate CSV files in the appropriate directory.

        Args:           
            dataframes (List[pd.DataFrame]): The list of dataframes to save.
            file_names (List[str]): The list of file names to save the dataframes to.
            cumulative (bool): Whether to save to the cumulative scraped data directory (True) or the newly scraped data directory (False).
        """
        pass

    @abstractmethod
    def load_scraped_data(self, cumulative: bool = False) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Load the scraped data from a storage medium.

        Args:
            cumulative (bool): Whether to load the newly scraped data or the cumulative scraped data.

        Returns:
            List[pd.DataFrame]: List of DataFrames containing the loaded data.
        """
        pass

    @abstractmethod
    def save_column_mapping(self, column_mapping: Dict[str, str], file_name: str) -> bool:
        """
        Save the column mapping to a json file.

        Args:
            column_mapping (Dict[str, str]): The column mapping to save.
            file_name (str): The name of the file to save the column mapping to.
        """
        pass

    @abstractmethod
    def load_dataframe(self, file_name: str) -> pd.DataFrame:
        """
        Load the indicated dataframe from the appropriate directory.

        Returns:
            pd.DataFrame: The indicated dataframe.
        """
        pass
