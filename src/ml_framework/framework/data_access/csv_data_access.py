"""
csv_data_access.py

Concrete implementation of the BaseDataAccess class for saving and loading data from CSV files.

This class isolates the data access layer from the rest of the application, allowing data to be saved and loaded from 
different sources (e.g., CSV files, databases, APIs) without changing the rest of the application.
"""

import pandas as pd
import logging
from typing import List, Tuple, Dict
from pathlib import Path
import json

from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.framework.data_access.base_data_access import BaseDataAccess
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.app_file_handling.base_app_file_handler import BaseAppFileHandler
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler

class CSVDataAccess(BaseDataAccess):
    def __init__(self, config: BaseConfigManager, app_logger: BaseAppLogger, 
                 app_file_handler: BaseAppFileHandler, error_handler: BaseErrorHandler):
        """
        Initialize the DataAccess object with the given configuration and logger.

        Args:
            config (BaseConfigManager): The configuration object.
            logger (BaseAppLogger): The logger object.
            file_handler (BaseFileHandler): The file handler object.

        Raises:
            ConfigurationError: If there's an issue with the configuration.
        """
        try:
            self.config = config
            self.app_logger = app_logger
            self.app_file_handler = app_file_handler
            self.error_handler = error_handler

        except AttributeError as e:
            raise self.error_handler.create_error_handler(
                'configuration',
                f"Missing required configuration: {str(e)}"
            )
        
    @staticmethod
    def log_performance(func):
        """Decorator factory for performance logging"""
        def wrapper(*args, **kwargs):
            # Get the self instance from args since this is now a static method
            instance = args[0]
            return instance.app_logger.log_performance(func)(*args, **kwargs)
        return wrapper

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
                save_path = self._get_save_directory(cumulative, file_name)
                self.app_file_handler.ensure_directory(save_path)
                self.app_file_handler.write_csv(df, save_path / file_name)
            self.app_logger.structured_log(logging.INFO, "Dataframes saved successfully")
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_storage',
                f"Error saving dataframes: {str(e)}"
            )

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
            self.app_logger.structured_log(logging.INFO, "Loading scraped data")
            scraped_path = self._get_load_directory(cumulative)
            all_dfs, file_names = self._load_dataframes(scraped_path)
            
            self.app_logger.structured_log(logging.INFO, "Data loaded successfully",
                           dataframe_count=len(all_dfs), scraped_path=str(scraped_path))
            
            return all_dfs, file_names

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_storage',
                f"Unexpected error loading scraped data: {str(e)}"
            )
        
    @log_performance
    def save_column_mapping(self, column_mapping: Dict[str, str], file_name: str) -> bool:
        """
        Save the column mapping to a json file with proper formatting.

        Args:
            column_mapping (Dict[str, str]): The column mapping to save.
            file_name (str): The name of the file to save the column mapping to.

        Returns:
            bool: True if the column mapping was saved successfully, False otherwise.
        """
        try:
            file_path = self._get_save_directory(cumulative=True, file_name=file_name)
            self.app_file_handler.write_json(column_mapping, file_path / file_name)
            self.app_logger.structured_log(logging.INFO, "Column mapping saved successfully",
                           file_path=str(file_path / file_name))
            return True
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_storage',
                f"Error saving column mapping: {str(e)}"
            )
        
    @log_performance    
    def load_dataframe(self, file_name: str) -> pd.DataFrame:
        """
        Load the data from a CSV file.

        Args:
            file_name (str): The name of the file to load the data from.

        Returns:
            pd.DataFrame: The processed dataframe.

        Raises:
            DataStorageError: If there's an error loading the data.
            DataValidationError: If the loaded data is invalid or inconsistent.
        """
        try:
            file_path = self._get_load_directory(cumulative=True, file_name=file_name)
            file_path = file_path / file_name
            if not file_path.exists():
                raise self.error_handler.create_error_handler(
                    'data_storage',
                    f"File {file_name} not found in {file_path}"
                )
            df = self.app_file_handler.read_csv(file_path)
            if df.empty:
                raise self.error_handler.create_error_handler(
                    'data_validation',
                    f"Loaded DataFrame from {file_name} is empty"
                )
            self.app_logger.structured_log(logging.INFO, "Data loaded successfully",
                           dataframe_count=len(df), file_path=str(file_path))
            return df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_storage',
                f"Unexpected error loading processed data: {str(e)}"
            )
        
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
            file_path = self._get_save_directory(cumulative, file_name)
            self.app_file_handler.write_csv(df, file_path / file_name)
            self.app_logger.structured_log(logging.INFO, "Data saved successfully",
                           file_path=str(file_path / file_name))
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_storage',
                f"Error saving data to {file_name}: {str(e)}"
            )

    @log_performance
    def _get_save_directory(self, cumulative: bool, file_name: str) -> Path:
        """
        Get the appropriate directory for saving data.

        Args:
            cumulative (bool): Whether to use the cumulative scraped data directory.

        Returns:
            Path: The directory path for saving data.

        Raises:
            DataStorageError: If the directory doesn't exist or can't be created.
        """

        match file_name:
            case self.config.column_mapping_file:
                save_path = Path(self.config.processed_data_directory)
            case self.config.team_centric_data_file:
                save_path = Path(self.config.processed_data_directory)
            case self.config.game_centric_data_file:
                save_path = Path(self.config.processed_data_directory)
            case self.config.engineered_data_file:
                save_path = Path(self.config.engineered_data_directory)
            case self.config.training_data_file:
                save_path = Path(self.config.training_data_directory)
            case self.config.validation_data_file:
                save_path = Path(self.config.training_data_directory)
            case self.config.evaluation_data_file:
                save_path = Path(self.config.evaluation_data_directory)
            case _ if file_name and "predictions" in file_name:
                save_path = Path(self.config.predictions_directory)
            case _:
                save_path = Path(self.config.cumulative_scraped_directory if cumulative else self.config.newly_scraped_directory)

        if not save_path.exists():
            raise self.error_handler.create_error_handler(
                'data_storage',
                f"Directory {save_path} not found"
            )
        
        return save_path

    def _get_load_directory(self, cumulative: bool, file_name: str = None) -> Path:
        """
        Get the appropriate directory for loading data.

        Args:
            cumulative (bool): Whether to use the cumulative scraped data directory.

        Returns:
            Path: The directory path for loading data.

        Raises:
            DataStorageError: If the directory doesn't exist.
        """
        self.app_logger.structured_log(logging.INFO, "Getting load directory",
                       cumulative=cumulative, file_name=str(file_name))
        
        match file_name:
            case self.config.column_mapping_file:
                load_path = Path(self.config.processed_data_directory)
            case self.config.team_centric_data_file:
                load_path = Path(self.config.processed_data_directory)
            case self.config.game_centric_data_file:
                load_path = Path(self.config.processed_data_directory)
            case self.config.engineered_data_file:
                load_path = Path(self.config.engineered_data_directory)
            case self.config.training_data_file:
                load_path = Path(self.config.training_data_directory)
            case self.config.validation_data_file:
                load_path = Path(self.config.training_data_directory)
            case self.config.evaluation_data_file:
                load_path = Path(self.config.evaluation_data_directory)
            case _ if file_name and "predictions" in file_name:
                load_path = Path(self.config.predictions_directory)
            case _:
                load_path = Path(self.config.cumulative_scraped_directory if cumulative else self.config.newly_scraped_directory)

        if not load_path.exists():
            raise self.error_handler.create_error_handler(
                'data_storage',
                f"Directory {load_path} not found"
            )
        
        return load_path

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
        self.app_logger.structured_log(logging.INFO, "Loading dataframes")
        all_dfs: List[pd.DataFrame] = []
        file_names = []
        for file in self.config.scraped_boxscore_files:
            file_path = scraped_path / file
            if not file_path.exists():
                raise self.error_handler.create_error_handler(
                    'data_storage',
                    f"File {file} not found in {scraped_path}"
                )
            file_names.append(file)
            try:
                df = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                self.app_logger.structured_log(logging.WARNING, "Empty CSV file encountered",
                             file_path=str(file_path))
                df = pd.DataFrame()  # Create empty DataFrame - some days no games are played so it might be empty
            except Exception as e:
                raise self.error_handler.create_error_handler(
                    'data_storage',
                    f"Error loading dataframe from {file}: {str(e)}"
                )

            all_dfs.append(df)
        self.app_logger.structured_log(logging.INFO, "Dataframes loaded successfully",
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
                raise self.error_handler.create_error_handler(
                    'data_validation',
                    "No dataframes loaded"
                )

            expected_columns = set(dataframes[0].columns)
            for i, df in enumerate(dataframes[1:], start=1):
                if set(df.columns) != expected_columns:
                    raise self.error_handler.create_error_handler(
                        'data_validation',
                        f"Dataframe {i} has inconsistent columns"
                    )

            self.app_logger.structured_log(logging.INFO, "Data validation completed successfully",
                           dataframe_count=len(dataframes))

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_validation',
                f"Unexpected error during data validation: {str(e)}"
            )