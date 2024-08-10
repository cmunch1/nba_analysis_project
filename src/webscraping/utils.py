"""
utils.py

This module provides utility functions for web scraping NBA data.
It includes functions for activating web drivers, determining scraping dates and seasons,
validating scraped data, and concatenating scraped datasets.

Key components:
- Web driver activation
- Scraping date and season determination
- Data validation
- Data concatenation
"""

import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from pytz import timezone
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from ..config.config import config
from ..data_access.data_access import DataAccess

data_access = DataAccess()

logger = logging.getLogger(__name__)



def get_start_date_and_seasons() -> Tuple[str, List[str]]:
    """
    Determine the start date and seasons for scraping.
    Example format: "10/1/2021", ["2023-24", "2022-23", "2021-22"]

    Returns:
        Tuple[str, List[str]]: A tuple containing the start date (str) and a list of seasons (List[str]).
    
    Raises:
        ValueError: If there are inconsistent dates in scraped data.
    """
    try:
        if config.full_scrape:
            seasons = [f"{season}-{str(season + 1)[-2:]}" for season in range(config.start_season, datetime.now().year)]
            first_start_date = f"{config.regular_season_start_month}/1/{config.start_season}"
        else:
            scraped_data = data_access.load_scraped_data(cumulative=True)
            first_start_date, seasons = determine_scrape_start(scraped_data)

            if first_start_date is None:
                logger.error("Previous scraped data has inconsistent dates")
                raise ValueError("Inconsistent dates in scraped data")

        logger.info(f"Start date: {first_start_date}, Seasons: {seasons}")
        return first_start_date, seasons
    except Exception as e:
        logger.error(f"Error in get_start_date_and_seasons: {str(e)}")
        raise

def determine_scrape_start(scraped_data: List[pd.DataFrame]) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Determine where to begin scraping for more games based on the latest game in the dataset.

    Args:
        scraped_data (List[pd.DataFrame]): List of DataFrames that have been scraped.

    Returns:
        Tuple[Optional[str], Optional[List[str]]]: A tuple containing the start date (str) and a list of seasons (List[str]).
        Returns (None, None) if there's an error.
    """
    try:
        last_date = None
        for i, df in enumerate(scraped_data):
            if df.empty:
                logger.error(f"Dataframe {i} is empty")
                return None, None
            date_col = next((col for col in config.game_date_header_variations if col in df.columns), None)
            if date_col is None:
                logger.error(f"Dataframe {i} does not have a recognized game date column")
                return None, None
            
            df[date_col] = pd.to_datetime(df[date_col])
            current_last_date = df[date_col].max()
            
            if last_date is None:
                last_date = current_last_date
            elif current_last_date != last_date:
                logger.error(f"Dataframe {i} has a different last date than the first dataframe")
                return None, None

        last_season = last_date.year if last_date.month >= 10 else last_date.year - 1
        start_date = (last_date + timedelta(days=1)).strftime("%m/%d/%Y")

        today = datetime.now(timezone('EST'))
        current_season = today.year if today.month >= 10 else today.year - 1

        seasons = [f"{season}-{str(season + 1)[-2:]}" for season in range(last_season, current_season + 1)]

        logger.info(f"Last date in dataset: {last_date}")
        logger.info(f"Last season in dataset: {last_season}")
        logger.info(f"Current season: {current_season}")
        logger.info(f"Seasons to scrape: {seasons}")
        logger.info(f"Start date: {start_date}")

        return start_date, seasons
    except Exception as e:
        logger.error(f"Error in determine_scrape_start: {str(e)}")
        return None, None

def validate_data(cumulative: bool = False) -> None:
    """
    Validate the boxscores dataframe.

    Args:
        cumulative (bool): Whether to validate cumulative data or not.
    """
    try:
        scraped_data = data_access.load_scraped_data(cumulative)
        response = validate_scraped_dataframes(scraped_data)

        if response == "Pass":
            logger.info("All scraped dataframes are consistent")
        else:
            logger.error(f"Scraped dataframes are inconsistent: {response}")
    except Exception as e:
        logger.error(f"Error in validate_data: {str(e)}")

def validate_scraped_dataframes(scraped_dataframes: List[pd.DataFrame]) -> str:
    """
    Validate the consistency of scraped dataframes.

    Args:
        scraped_dataframes (List[pd.DataFrame]): List of scraped dataframes to validate.

    Returns:
        str: "Pass" if all dataframes are consistent, otherwise an error message.
    """
    try:
        num_rows = 0
        game_ids = None
        
        for i, df in enumerate(scraped_dataframes):
            if df.duplicated().any():
                return f"Dataframe {i} has duplicate records"
            
            if df.isnull().values.any():
                return f"Dataframe {i} has null values"

            df = df.sort_values(by='GAME_ID')

            if i == 0:
                num_rows = df.shape[0]
                game_ids = df['GAME_ID']
            else:
                if num_rows != df.shape[0]:
                    return f"Dataframe {i} does not match the number of rows of the first dataframe"
                
                if not np.array_equal(game_ids.values, df['GAME_ID'].values):
                    return f"Dataframe {i} does not match the game ids of the first dataframe"
        
        return "Pass"
    except Exception as e:
        logger.error(f"Error in validate_scraped_dataframes: {str(e)}")
        return f"Error during validation: {str(e)}"

def concatenate_scraped_data() -> None:
    """
    Concatenate newly scraped data with cumulative scraped data.
    """
    try:
        newly_scraped = data_access.load_scraped_data(cumulative=False)
        cumulative_scraped = data_access.load_scraped_data(cumulative=True)

        if not newly_scraped or not cumulative_scraped:
            logger.error("Either newly scraped or cumulative data is missing")
            return

        # Check for empty dataframes in newly_scraped
        empty_dfs = [i for i, df in enumerate(newly_scraped) if df.empty]
        if empty_dfs:
            logger.warning(f"Empty dataframes found in newly scraped data at indices: {empty_dfs}")

        for i, (new_df, cum_df, file_name) in enumerate(zip(newly_scraped, cumulative_scraped, config.scraped_boxscore_files)):
            if new_df.empty:
                logger.warning(f"Skipping empty dataframe for {file_name}")
                continue

            combined_df = pd.concat([cum_df, new_df], ignore_index=True)
            combined_df = combined_df.sort_values(by=['GAME_ID', 'TEAM_ID'])
            combined_df = combined_df.drop_duplicates(subset=['GAME_ID', 'TEAM_ID'], keep='last')
            
            data_access.save_scraped_data(combined_df, file_name, cumulative=True)
            logger.info(f"Successfully concatenated and saved {file_name}")

        logger.info("Completed concatenation of all scraped data files")
    except Exception as e:
        logger.error(f"Error in concatenate_scraped_data: {str(e)}")


