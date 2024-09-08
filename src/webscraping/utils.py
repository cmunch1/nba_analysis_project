"""
utils.py

This module provides utility functions for web scraping NBA data.
It includes functions for determining scraping dates and seasons,


Key components:
- Scraping date and season determination


Functions:
- get_start_date_and_seasons: Determine the start date and seasons for scraping
- determine_scrape_start: Determine where to begin scraping based on existing data
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
            scraped_data, file_names = data_access.load_scraped_data(cumulative=True)
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


