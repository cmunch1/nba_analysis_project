"""
utils.py

This module provides utility functions for web scraping NBA data.
It includes functions for determining scraping dates and seasons,
validating scraped data, and concatenating scraped datasets.

Key components:
- Scraping date and season determination
- Data validation
- Data concatenation

Functions:
- get_start_date_and_seasons: Determine the start date and seasons for scraping
- determine_scrape_start: Determine where to begin scraping based on existing data
- validate_data: Validate the boxscores dataframe
- validate_scraped_dataframes: Validate the consistency of scraped dataframes
- concatenate_scraped_data: Concatenate newly scraped data with cumulative scraped data
"""

import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from pytz import timezone

from ..config.abstract_config import AbstractConfig
from ..data_access.abstract_data_access import AbstractDataAccess
from ..error_handling.custom_exceptions import (
    DataValidationError, DataProcessingError, ConfigurationError
)

logger = logging.getLogger(__name__)

def get_start_date_and_seasons(config: AbstractConfig, data_access: AbstractDataAccess) -> Tuple[str, List[str]]:
    """
    Determine the start date and seasons for scraping.
    Example format: "10/1/2021", ["2023-24", "2022-23", "2021-22"]

    Args:
        config (AbstractConfig): The configuration object.
        data_access (AbstractDataAccess): The data access object.

    Returns:
        Tuple[str, List[str]]: A tuple containing the start date (str) and a list of seasons (List[str]).
    
    Raises:
        ConfigurationError: If there's an issue with the configuration.
        DataValidationError: If there are inconsistent dates in scraped data.
        DataProcessingError: If there's an error processing the data.
    """
    try:
        if config.full_scrape:
            seasons = [f"{season}-{str(season + 1)[-2:]}" for season in range(config.start_season, datetime.now().year)]
            first_start_date = f"{config.regular_season_start_month}/1/{config.start_season}"
        else:
            scraped_data = data_access.load_scraped_data(cumulative=True)
            first_start_date, seasons = determine_scrape_start(scraped_data, config)

            if first_start_date is None and seasons is None:
                raise DataValidationError("Previous scraped data has inconsistent dates")
            if first_start_date is None:
                raise DataProcessingError("Error in determining scrape start date")

        logger.info(f"Start date: {first_start_date}, Seasons: {seasons}")
        return first_start_date, seasons
    except AttributeError as e:
        raise ConfigurationError(f"Missing required configuration: {str(e)}")
    except Exception as e:
        raise DataProcessingError(f"Error in get_start_date_and_seasons: {str(e)}")

def determine_scrape_start(scraped_data: List[pd.DataFrame], config: AbstractConfig) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Determine where to begin scraping for more games based on the latest game in the dataset.

    Args:
        scraped_data (List[pd.DataFrame]): List of DataFrames that have been scraped.
        config (AbstractConfig): The configuration object.

    Returns:
        Tuple[Optional[str], Optional[List[str]]]: A tuple containing the start date (str) and a list of seasons (List[str]).

    Raises:
        DataValidationError: If there are inconsistencies in the scraped data.
        DataProcessingError: If there's an error processing the data.
    """
    try:
        last_date = None
        for i, df in enumerate(scraped_data):
            if df.empty:
                raise DataValidationError(f"Dataframe {i} is empty")
            date_col = next((col for col in config.game_date_header_variations if col in df.columns), None)
            if date_col is None:
                raise DataValidationError(f"Dataframe {i} does not have a recognized game date column")
            
            df[date_col] = pd.to_datetime(df[date_col])
            current_last_date = df[date_col].max()
            
            if last_date is None:
                last_date = current_last_date
            elif current_last_date != last_date:
                raise DataValidationError(f"Dataframe {i} has a different last date than the first dataframe")

        # Calculate the last season based on the last date
        # If the month is October or later, it's considered part of the next season
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
    except DataValidationError:
        raise
    except Exception as e:
        raise DataProcessingError(f"Error in determine_scrape_start: {str(e)}")

def validate_data(config: AbstractConfig, data_access: AbstractDataAccess, cumulative: bool = False) -> None:
    """
    Validate the boxscores dataframe.

    Args:
        config (AbstractConfig): The configuration object.
        data_access (AbstractDataAccess): The data access object.
        cumulative (bool): Whether to validate cumulative data or not.

    Raises:
        DataValidationError: If the scraped dataframes are inconsistent.
        DataProcessingError: If there's an error during the validation process.
    """
    try:
        scraped_data = data_access.load_scraped_data(cumulative)
        response = validate_scraped_dataframes(scraped_data)

        if response == "Pass":
            logger.info("All scraped dataframes are consistent")
        else:
            raise DataValidationError(f"Scraped dataframes are inconsistent: {response}")
    except DataValidationError:
        raise
    except Exception as e:
        raise DataProcessingError(f"Error in validate_data: {str(e)}")

def validate_scraped_dataframes(scraped_dataframes: List[pd.DataFrame]) -> str:
    """
    Validate the consistency of scraped dataframes.

    Args:
        scraped_dataframes (List[pd.DataFrame]): List of scraped dataframes to validate.

    Returns:
        str: "Pass" if all dataframes are consistent, otherwise an error message.

    Raises:
        DataValidationError: If there are inconsistencies in the scraped dataframes.
    """
    try:
        num_rows = 0
        game_ids = None
        
        for i, df in enumerate(scraped_dataframes):
            if df.duplicated().any():
                raise DataValidationError(f"Dataframe {i} has duplicate records")
            
            if df.isnull().values.any():
                raise DataValidationError(f"Dataframe {i} has null values")

            df = df.sort_values(by='GAME_ID')

            if i == 0:
                num_rows = df.shape[0]
                game_ids = df['GAME_ID']
            else:
                if num_rows != df.shape[0]:
                    raise DataValidationError(f"Dataframe {i} does not match the number of rows of the first dataframe")
                
                if not np.array_equal(game_ids.values, df['GAME_ID'].values):
                    raise DataValidationError(f"Dataframe {i} does not match the game ids of the first dataframe")
        
        return "Pass"
    except DataValidationError:
        raise
    except Exception as e:
        raise DataProcessingError(f"Error in validate_scraped_dataframes: {str(e)}")

def concatenate_scraped_data(config: AbstractConfig, data_access: AbstractDataAccess) -> None:
    """
    Concatenate newly scraped data with cumulative scraped data.

    Args:
        config (AbstractConfig): The configuration object.
        data_access (AbstractDataAccess): The data access object.

    Raises:
        DataValidationError: If there are issues with the scraped data.
        DataProcessingError: If there's an error during the concatenation process.
    """
    try:
        newly_scraped = data_access.load_scraped_data(cumulative=False)
        cumulative_scraped = data_access.load_scraped_data(cumulative=True)

        if not newly_scraped or not cumulative_scraped:
            raise DataValidationError("Either newly scraped or cumulative data is missing")

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
            # Remove duplicates, keeping the last occurrence (which should be the most recent data)
            combined_df = combined_df.drop_duplicates(subset=['GAME_ID', 'TEAM_ID'], keep='last')
            
            data_access.save_scraped_data(combined_df, file_name, cumulative=True)
            logger.info(f"Successfully concatenated and saved {file_name}")

        logger.info("Completed concatenation of all scraped data files")
    except DataValidationError:
        raise
    except Exception as e:
        raise DataProcessingError(f"Error in concatenate_scraped_data: {str(e)}")