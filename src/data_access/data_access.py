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

class DataAccess(AbstractDataAccess):
    def __init__(self, config: AbstractConfig):
        self.config = config
        logging.basicConfig(level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)

    def save_scraped_data(self, df: pd.DataFrame, file_name: str, cumulative: bool = False) -> None:
        """
        Saves the dataframe to a csv file in the appropriate directory

        Args:
            df (pd.DataFrame): the scraped data to save
            file_name (str): the name of the file to save the data to
            cumulative (bool): whether to save to newly scraped data directory or the cumulative scraped data directory
        """ 
        if cumulative:
            if not Path(self.config.cumulative_scraped_directory).exists():
                self.logger.error("Could not find directory for cumulative scraped data")
                raise FileNotFoundError("Cumulative scraped data directory not found")
            file_path = Path(self.config.cumulative_scraped_directory)
        else:
            if not Path(self.config.newly_scraped_directory).exists():
                self.logger.warning("Could not find directory for newly scraped data")
                self.logger.info("Creating directory for newly scraped data")
                self.config.newly_scraped_path.mkdir(parents=True, exist_ok=True)
            file_path = Path(self.config.newly_scraped_directory)

        if file_name == "matchups":
            file_name = self.config.todays_matchups_file

        if file_name == "games_ids":
            file_name = self.config.todays_games_ids_file
            
        df.to_csv(file_path / file_name, index=False)
        self.logger.info(f"Data saved to {file_path / file_name}")

    def load_scraped_data(self, cumulative: bool = False) -> List[pd.DataFrame]:
        """
        Get the scraped data from the csv files, either the newly scraped data or the cumulative scraped data
        Retrieves all the scraped data from the csv files and returns them as a list of DataFrames

        Args:
            cumulative (bool): whether to load the newly scraped data or the cumulative scraped data

        Returns:
            List[pd.DataFrame]: list of DataFrames
        """ 
        scraped_path = Path(self.config.cumulative_scraped_directory) if cumulative else Path(self.config.newly_scraped_directory)

        all_dfs: List[pd.DataFrame] = []

        if not scraped_path.exists():
            self.logger.error(f"Directory {scraped_path} not found")
            raise FileNotFoundError(f"Directory {scraped_path} not found")
    
        for file in self.config.scraped_boxscore_files:
            if not (scraped_path / file).exists():
                self.logger.error(f"File {file} not found in {scraped_path}")
                raise FileNotFoundError(f"File {file} not found in {scraped_path}")
            
            df = pd.read_csv(scraped_path / file)
            all_dfs.append(df)

        self.logger.info(f"Loaded {len(all_dfs)} dataframes from {scraped_path}")
        return all_dfs