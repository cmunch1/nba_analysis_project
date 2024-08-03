"""
data_access.py

Wrapper class for saving and loading data. 

Isolates the data access layer from the rest of the application so that the data can be saved and loaded from 
different sources (e.g. csv files, databases, APIs) without changing the rest of the application.
"""

import pandas as pd
from pathlib import Path
import logging
from typing import List

from config import config

class DataAccess:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def save_scraped_data(self, df: pd.DataFrame, file_name: str, cumulative: bool = False) -> None:
        """
        Saves the dataframe to a csv file in the appropriate directory

        Args:
            df (pd.DataFrame): the scraped data to save
            file_name (str): the name of the file to save the data to
            cumulative (bool): whether to save to newly scraped data directory or the cumulative scraped data directory
        """ 
        if cumulative:
            if not config.cumulative_scraped_path.exists():
                self.logger.error("Could not find directory for cumulative scraped data")
                raise FileNotFoundError("Cumulative scraped data directory not found")
            file_path = config.cumulative_scraped_path 
        else:
            if not config.newly_scraped_path.exists():
                self.logger.warning("Could not find directory for newly scraped data")
                self.logger.info("Creating directory for newly scraped data")
                config.newly_scraped_path.mkdir(parents=True, exist_ok=True)
            file_path = config.newly_scraped_path

        if file_name == "matchups":
            file_name = config.todays_matchups_file

        if file_name == "games_ids":
            file_name = config.todays_games_ids_file
            
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
        scraped_path = config.cumulative_scraped_path if cumulative else config.newly_scraped_path

        all_dfs: List[pd.DataFrame] = []

        if not scraped_path.exists():
            self.logger.error(f"Directory {scraped_path} not found")
            raise FileNotFoundError(f"Directory {scraped_path} not found")
    
        for file in config.scraped_boxscore_files:
            if not (scraped_path / file).exists():
                self.logger.error(f"File {file} not found in {scraped_path}")
                raise FileNotFoundError(f"File {file} not found in {scraped_path}")
            
            df = pd.read_csv(scraped_path / file)
            all_dfs.append(df)

        self.logger.info(f"Loaded {len(all_dfs)} dataframes from {scraped_path}")
        return all_dfs
