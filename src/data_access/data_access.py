"""
data_access.py

Concrete implementation of the AbstractDataAccess class for saving and loading data from CSV files.

This class isolates the data access layer from the rest of the application, allowing data to be saved and loaded from 
different sources (e.g., CSV files, databases, APIs) without changing the rest of the application.
"""

import pandas as pd
import logging
from typing import List, Tuple
from pathlib import Path

from ..config.abstract_config import AbstractConfig
from .abstract_data_access import AbstractDataAccess
from ..error_handling.custom_exceptions import (
    DataStorageError, ConfigurationError, DataValidationError
)
from ..logging.logging_utils import log_performance, structured_log

logger = logging.getLogger(__name__)

class DataAccess(AbstractDataAccess):
    @log_performance
    def __init__(self, config: AbstractConfig):
        """
        Initialize the DataAccess object with the given configuration.

        Args:
            config (AbstractConfig): The configuration object.

        Raises:
            ConfigurationError: If there's an issue with the configuration.
        """
        try:
            self.config = config
            structured_log(logger, logging.INFO, "DataAccess initialized successfully",
                           config_type=type(config).__name__)
        except AttributeError as e:
            raise ConfigurationError(f"Missing required configuration: {str(e)}")

    @log_performance
    def _save_dataframe_csv(self, df: pd.DataFrame, file_name: str, cumulative: bool = False) -> None:
        """
        Save the dataframe to a CSV file in the appropriate directory.

        Args:
            df (pd.DataFrame): The scraped data to save.
            file_name (str): The name of the file to save the data to.
            cumulative (bool): Whether to save to the cumulative scraped data directory (True) or the newly scraped data directory (False).

        Raises:
            DataStorageError: If there's an error saving the data.
        """
        try:
            file_path = self._get_save_directory(cumulative)
                        
            df.to_csv(file_path / file_name, index=False)
            structured_log(logger, logging.INFO, "Data saved successfully",
                           file_path=str(file_path / file_name))
        except Exception as e:
            raise DataStorageError(f"Error saving data to {file_name}: {str(e)}")
        
    @log_performance
    def save_dataframes(self, dataframes: List[pd.DataFrame], file_names: List[str], cumulative: bool = False) -> None:
        """
        Save the list of dataframes to separate CSV files in the appropriate directory.

        Args:           
            dataframes (List[pd.DataFrame]): The list of dataframes to save.
            file_names (List[str]): The list of file names to save the dataframes to.
            cumulative (bool): Whether to save to the cumulative scraped data directory (True) or the newly scraped data directory (False).
        """         
        try:
            for df, file_name in zip(dataframes, file_names):
                self._save_dataframe_csv(df, file_name, cumulative=cumulative)
            structured_log(logger, logging.INFO, "Dataframes saved successfully")
        except Exception as e:
            raise DataStorageError(f"Error saving dataframes: {str(e)}")

    @log_performance
    def load_scraped_data(self, cumulative: bool = False) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Load the scraped data from CSV files, either the newly scraped data or the cumulative scraped data.

        Args:
            cumulative (bool): Whether to load the cumulative scraped data (True) or the newly scraped data (False).

        Returns:
            List[pd.DataFrame]: List of loaded DataFrames.

        Raises:
            DataStorageError: If there's an error loading the data.
            DataValidationError: If the loaded data is invalid or inconsistent.
        """
        try:
            scraped_path = self._get_load_directory(cumulative)
            all_dfs, file_names = self._load_dataframes(scraped_path)
            self._validate_loaded_data(all_dfs)
            
            structured_log(logger, logging.INFO, "Data loaded successfully",
                           dataframe_count=len(all_dfs), scraped_path=str(scraped_path))
            
            return all_dfs, file_names

        except (DataStorageError, DataValidationError):
            raise
        except Exception as e:
            raise DataStorageError(f"Unexpected error loading scraped data: {str(e)}")

    @log_performance
    def _get_save_directory(self, cumulative: bool) -> Path:
        """
        Get the appropriate directory for saving data.

        Args:
            cumulative (bool): Whether to use the cumulative scraped data directory.

        Returns:
            Path: The directory path for saving data.

        Raises:
            DataStorageError: If the directory doesn't exist or can't be created.
        """
        if cumulative:
            if not Path(self.config.cumulative_scraped_directory).exists():
                raise DataStorageError("Could not find directory for cumulative scraped data")
            return Path(self.config.cumulative_scraped_directory)
        else:
            newly_scraped_dir = Path(self.config.newly_scraped_directory)
            if not newly_scraped_dir.exists():
                structured_log(logger, logging.WARNING, "Directory for newly scraped data not found")
                structured_log(logger, logging.INFO, "Creating directory for newly scraped data")
                newly_scraped_dir.mkdir(parents=True, exist_ok=True)
            return newly_scraped_dir


    def _get_load_directory(self, cumulative: bool) -> Path:
        """
        Get the appropriate directory for loading data.

        Args:
            cumulative (bool): Whether to use the cumulative scraped data directory.

        Returns:
            Path: The directory path for loading data.

        Raises:
            DataStorageError: If the directory doesn't exist.
        """
        scraped_path = Path(self.config.cumulative_scraped_directory if cumulative else self.config.newly_scraped_directory)
        if not scraped_path.exists():
            raise DataStorageError(f"Directory {scraped_path} not found")
        return scraped_path

    @log_performance
    def _load_dataframes(self, scraped_path: Path) -> List[pd.DataFrame]:
        """
        Load dataframes from CSV files in the specified directory.

        Args:
            scraped_path (Path): The directory to load CSV files from.

        Returns:
            List[pd.DataFrame]: List of loaded DataFrames.

        Raises:
            DataStorageError: If a file is not found or can't be loaded.
            DataValidationError: If a loaded DataFrame is empty.
        """
        all_dfs: List[pd.DataFrame] = []
        file_names = []
        for file in self.config.scraped_boxscore_files:
            file_path = scraped_path / file
            if not file_path.exists():
                raise DataStorageError(f"File {file} not found in {scraped_path}")
            file_names.append(file)
            df = pd.read_csv(file_path)
            if df.empty:
                raise DataValidationError(f"Loaded DataFrame from {file} is empty")
            all_dfs.append(df)
        structured_log(logger, logging.INFO, "Dataframes loaded successfully",
                       dataframe_count=len(all_dfs), scraped_path=str(scraped_path))
        return all_dfs, file_names

    @log_performance
    def _validate_loaded_data(self, dataframes: List[pd.DataFrame]) -> None:
        """
        Validate the loaded dataframes for consistency.

        Args:
            dataframes (List[pd.DataFrame]): List of loaded dataframes.

        Raises:
            DataValidationError: If the loaded data is invalid or inconsistent.
        """
        try:
            if not dataframes:
                raise DataValidationError("No dataframes loaded")

            expected_columns = set(dataframes[0].columns)
            for i, df in enumerate(dataframes[1:], start=1):
                if set(df.columns) != expected_columns:
                    raise DataValidationError(f"Dataframe {i} has inconsistent columns")

            structured_log(logger, logging.INFO, "Data validation completed successfully",
                           dataframe_count=len(dataframes))

        except DataValidationError:
            raise
        except Exception as e:
            raise DataValidationError(f"Unexpected error during data validation: {str(e)}")