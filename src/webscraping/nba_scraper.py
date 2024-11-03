"""
nba_scraper.py

This module provides a facade for scraping NBA data, including boxscores and schedules.
It combines the functionality of BoxscoreScraper and ScheduleScraper into a single interface,
making it easier to manage web scraping tasks for NBA data. The module uses custom exceptions
for more specific error handling and implements enhanced logging for better debugging and monitoring.

Key Classes:
    - NbaScraper: Main class that orchestrates NBA data scraping operations.

Dependencies:
    - AbstractBoxscoreScraper and AbstractScheduleScraper from abstract_scraper_classes module
    - Custom exceptions from error_handling module
    - Logging utilities from logging_utils module
"""

from typing import List
import logging
import re

from .abstract_scraper_classes import (
    AbstractNbaScraper,
    AbstractBoxscoreScraper, 
    AbstractScheduleScraper,
)
from ..config.config import AbstractConfig
from ..error_handling.custom_exceptions import (
    ConfigurationError,
    ScrapingError,
    DataValidationError,
    DataStorageError
)
from ..logging.logging_utils import log_performance, log_context, structured_log

logger = logging.getLogger(__name__)

class NbaScraper(AbstractNbaScraper):
    """
    A facade class that combines boxscore and schedule scraping functionality for NBA data.

    This class delegates scraping tasks to specialized scraper classes.

    Attributes:
        _config (AbstractConfig): Configuration object.
        _boxscore_scraper (AbstractBoxscoreScraper): An instance of BoxscoreScraper.
        _schedule_scraper (AbstractScheduleScraper): An instance of ScheduleScraper.
    """

    @log_performance
    def __init__(self, config: AbstractConfig, boxscore_scraper: AbstractBoxscoreScraper, schedule_scraper: AbstractScheduleScraper):
        """
        Initialize the NbaScraper with configuration and scraper instances.

        Args:
            config (AbstractConfig): Configuration object.
            boxscore_scraper (AbstractBoxscoreScraper): BoxscoreScraper instance.
            schedule_scraper (AbstractScheduleScraper): ScheduleScraper instance.

        Raises:
            ConfigurationError: If there's an issue with the provided configuration or scraper instances.
        """
        try:
            self._config = config
            self._boxscore_scraper = boxscore_scraper
            self._schedule_scraper = schedule_scraper

            if not isinstance(boxscore_scraper, AbstractBoxscoreScraper):
                raise ConfigurationError("Invalid boxscore_scraper instance")
            if not isinstance(schedule_scraper, AbstractScheduleScraper):
                raise ConfigurationError("Invalid schedule_scraper instance")

            structured_log(logger, logging.INFO, "NbaScraper initialized successfully", 
                           boxscore_scraper_type=type(boxscore_scraper).__name__,
                           schedule_scraper_type=type(schedule_scraper).__name__)
        except Exception as e:
            structured_log(logger, logging.ERROR, "Error initializing NbaScraper", 
                           error_message=str(e),
                           error_type=type(e).__name__)
            raise ConfigurationError(f"Error initializing NbaScraper: {str(e)}")

    @log_performance
    def scrape_and_save_all_boxscores(self, seasons: List[str], first_start_date: str) -> None:
        """
        Scrape and save all boxscores for the given seasons.

        Args:
            seasons (List[str]): A list of seasons to scrape (e.g., ["2021-22", "2022-23"]).
            first_start_date (str): The start date for the first season in MM/DD/YYYY format.

        Raises:
            DataValidationError: If the input parameters are invalid.
            ScrapingError: If there's an error during the scraping process.
            DataStorageError: If there's an error saving the scraped data.
        """
        try:
            self._validate_boxscore_input(seasons, first_start_date)

            with log_context(operation="scrape_boxscores", seasons=seasons, start_date=first_start_date):
                structured_log(logger, logging.INFO, "Starting to scrape boxscores", 
                               seasons=seasons, 
                               start_date=first_start_date)
                
                self._boxscore_scraper.scrape_and_save_all_boxscores(seasons, first_start_date)
                
                structured_log(logger, logging.INFO, "Boxscore scraping completed successfully")
        except (DataValidationError, ScrapingError, DataStorageError) as e:
            structured_log(logger, logging.ERROR, f"{type(e).__name__} in scrape_and_save_all_boxscores", 
                           error_message=str(e),
                           error_type=type(e).__name__)
            raise
        except Exception as e:
            structured_log(logger, logging.ERROR, "Unexpected error in scrape_and_save_all_boxscores", 
                           error_message=str(e),
                           error_type=type(e).__name__)
            raise ScrapingError(f"Unexpected error occurred while scraping boxscores: {str(e)}")

    @log_performance
    def scrape_and_save_matchups_for_day(self, search_day: str) -> bool:    
        """
        Scrape and save matchups for a specific day.

        Args:
            search_day (str): The day to search for matchups (3-letter abbreviation, e.g., 'MON', 'TUE').

        Returns:
            bool: True if matchups were found and saved, False otherwise.

        Raises:
            DataValidationError: If the search_day parameter is invalid.
            ScrapingError: If there's an error during the scraping process.
            DataStorageError: If there's an error saving the scraped data.
        """
        try:
            self._validate_search_day(search_day)

            with log_context(operation="scrape_matchups", search_day=search_day):
                structured_log(logger, logging.INFO, "Starting to scrape matchups", 
                               search_day=search_day)
                
                if self._schedule_scraper.scrape_and_save_matchups_for_day(search_day):
                    structured_log(logger, logging.INFO, "Matchup scraping completed successfully")
                    return True
                else:
                    structured_log(logger, logging.INFO, "No matchups found for the given day")
                    return False

        except (DataValidationError, ScrapingError, DataStorageError) as e:
            structured_log(logger, logging.ERROR, f"{type(e).__name__} in scrape_and_save_matchups_for_day", 
                           error_message=str(e),
                           error_type=type(e).__name__)
            raise
        except Exception as e:
            structured_log(logger, logging.ERROR, "Unexpected error in scrape_and_save_matchups_for_day", 
                           error_message=str(e),
                           error_type=type(e).__name__)
            raise ScrapingError(f"Unexpected error occurred while scraping matchups: {str(e)}")

    def _validate_boxscore_input(self, seasons: List[str], first_start_date: str) -> None:
        """
        Validate input parameters for scraping boxscores.

        Args:
            seasons (List[str]): A list of seasons to scrape.
            first_start_date (str): The start date for the first season.

        Raises:
            DataValidationError: If the input parameters are invalid.
        """
        if not seasons:
            raise DataValidationError("Seasons list cannot be empty")
        
        # Check date format MM/DD/YYYY
        if not isinstance(first_start_date, str) or not re.match(r'^\d{2}/\d{2}/\d{4}$', first_start_date):
            raise DataValidationError("Invalid first_start_date format. Expected MM/DD/YYYY")

    def _validate_search_day(self, search_day: str) -> None:
        """
        Validate the search_day parameter for scraping matchups.

        Args:
            search_day (str): The day to search for matchups.

        Raises:
            DataValidationError: If the search_day parameter is invalid.
        """
        if not isinstance(search_day, str) or len(search_day) != 3:
            raise DataValidationError("Invalid search_day format. Expected 3-letter day abbreviation (e.g., 'MON', 'TUE')")