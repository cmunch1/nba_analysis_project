"""
data_access.py

Concrete implementation of the AbstractDataAccess class for saving and loading data from CSV files.

This class isolates the data access layer from the rest of the application so that the data can be saved and loaded from 
different sources (e.g. csv files, databases, APIs) without changing the rest of the application.
"""

import pandas as pd
import logging
from typing import List
from pathlib import Path

from ..config.abstract_config import AbstractConfig
from .abstract_data_access import AbstractDataAccess
from ..error_handling.custom_exceptions import (
    DataStorageError, ConfigurationError, DataValidationError
)

class DataAccess(AbstractDataAccess):
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
            logging.basicConfig(level=getattr(logging, self.config.log_level),
                format='%(asctime)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(self.__class__.__name__)
        except AttributeError as e:
            raise ConfigurationError(f"Missing required configuration: {str(e)}")

    def save_scraped_data(self, df: pd.DataFrame, file_name: str, cumulative: bool = False) -> None:
        """
        Saves the dataframe to a csv file in the appropriate directory.

        Args:
            df (pd.DataFrame): the scraped data to save
            file_name (str): the name of the file to save the data to
            cumulative (bool): whether to save to newly scraped data directory or the cumulative scraped data directory

        Raises:
            DataStorageError: If there's an error saving the data.
        """
        try:
            if cumulative:
                if not Path(self.config.cumulative_scraped_directory).exists():
                    raise DataStorageError("Could not find directory for cumulative scraped data")
                file_path = Path(self.config.cumulative_scraped_directory)
            else:
                if not Path(self.config.newly_scraped_directory).exists():
                    self.logger.warning("Could not find directory for newly scraped data")
                    self.logger.info("Creating directory for newly scraped data")
                    Path(self.config.newly_scraped_directory).mkdir(parents=True, exist_ok=True)
                file_path = Path(self.config.newly_scraped_directory)

            if file_name == "matchups":
                file_name = self.config.todays_matchups_file
            elif file_name == "games_ids":
                file_name = self.config.todays_games_ids_file
                
            df.to_csv(file_path / file_name, index=False)
            self.logger.info(f"Data saved to {file_path / file_name}")
        except Exception as e:
            raise DataStorageError(f"Error saving data to {file_name}: {str(e)}")

    def load_scraped_data(self, cumulative: bool = False) -> List[pd.DataFrame]:
        """
        Get the scraped data from the csv files, either the newly scraped data or the cumulative scraped data.
        Retrieves all the scraped data from the csv files and returns them as a list of DataFrames.

        Args:
            cumulative (bool): whether to load the newly scraped data or the cumulative scraped data

        Returns:
            List[pd.DataFrame]: list of DataFrames

        Raises:
            DataStorageError: If there's an error loading the data.
            DataValidationError: If the loaded data is invalid or inconsistent.
        """
        try:
            scraped_path = Path(self.config.cumulative_scraped_directory) if cumulative else Path(self.config.newly_scraped_directory)

            all_dfs: List[pd.DataFrame] = []

            if not scraped_path.exists():
                raise DataStorageError(f"Directory {scraped_path} not found")
        
            for file in self.config.scraped_boxscore_files:
                file_path = scraped_path / file
                if not file_path.exists():
                    raise DataStorageError(f"File {file} not found in {scraped_path}")
                
                df = pd.read_csv(file_path)
                if df.empty:
                    raise DataValidationError(f"Loaded DataFrame from {file} is empty")
                all_dfs.append(df)

            self.logger.info(f"Loaded {len(all_dfs)} dataframes from {scraped_path}")
            return all_dfs
        except (DataStorageError, DataValidationError):
            raise
        except Exception as e:
            raise DataStorageError(f"Unexpected error loading scraped data: {str(e)}")

    def validate_loaded_data(self, dataframes: List[pd.DataFrame]) -> None:
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

            # Add more validation checks as needed

        except DataValidationError:
            raise
        except Exception as e:
            raise DataValidationError(f"Unexpected error during data validation: {str(e)}")