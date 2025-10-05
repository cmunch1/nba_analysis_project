"""
nba_scraper.py

This module provides a facade for scraping NBA data, including boxscores and schedules.
It combines the functionality of BoxscoreScraper and ScheduleScraper into a single interface,
making it easier to manage web scraping tasks for NBA data. The module uses custom exceptions
for more specific error handling and implements enhanced logging for better debugging and monitoring.

Key Classes:
    - NbaScraper: Main class that orchestrates NBA data scraping operations.

Dependencies:
    - BaseBoxscoreScraper and BaseScheduleScraper from base_scraper_classes module
    - Custom exceptions from error_handling module
    - Logging utilities from logging_utils module
"""

from typing import List, Dict
import logging
import re

from .base_scraper_classes import (
    BaseNbaScraper,
    BaseBoxscoreScraper,
    BaseScheduleScraper,
)
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.error_handling.error_handler_factory import ErrorHandlerFactory
from ml_framework.core.app_logging import log_performance, log_context, structured_log, AppLogger

class NbaScraper(BaseNbaScraper):
    """
    A facade class that combines boxscore and schedule scraping functionality for NBA data.

    This class delegates scraping tasks to specialized scraper classes.

    Attributes:
        _config (BaseConfigManager): Configuration object.
        _boxscore_scraper (BaseBoxscoreScraper): An instance of BoxscoreScraper.
        _schedule_scraper (BaseScheduleScraper): An instance of ScheduleScraper.
    """

    @log_performance
    def __init__(self, config: BaseConfigManager, boxscore_scraper: BaseBoxscoreScraper, schedule_scraper: BaseScheduleScraper, app_logger: AppLogger, error_handler: ErrorHandlerFactory):
        """
        Initialize the NbaScraper with configuration and scraper instances.

        Args:
            config (BaseConfigManager): Configuration object.
            boxscore_scraper (BaseBoxscoreScraper): BoxscoreScraper instance.
            schedule_scraper (BaseScheduleScraper): ScheduleScraper instance.
            app_logger (AppLogger): Application logger instance.
            error_handler (ErrorHandlerFactory): Error handler factory instance.

        Raises:
            ConfigurationError: If there's an issue with the provided configuration or scraper instances.
        """
        try:
            self._config = config
            self._boxscore_scraper = boxscore_scraper
            self._schedule_scraper = schedule_scraper
            self.app_logger = app_logger
            self.error_handler = error_handler

            if not isinstance(boxscore_scraper, BaseBoxscoreScraper):
                raise error_handler.create_error_handler('configuration', "Invalid boxscore_scraper instance")
            if not isinstance(schedule_scraper, BaseScheduleScraper):
                raise error_handler.create_error_handler('configuration', "Invalid schedule_scraper instance")

            self.app_logger.structured_log(logging.INFO, "NbaScraper initialized successfully",
                           boxscore_scraper_type=type(boxscore_scraper).__name__,
                           schedule_scraper_type=type(schedule_scraper).__name__)
        except Exception as e:
            self.app_logger.structured_log(logging.ERROR, "Error initializing NbaScraper",
                           error_message=str(e),
                           error_type=type(e).__name__)
            raise error_handler.create_error_handler('configuration', f"Error initializing NbaScraper: {str(e)}")

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
                self.app_logger.structured_log( logging.INFO, "Starting to scrape boxscores", 
                               seasons=seasons, 
                               start_date=first_start_date)

                self._boxscore_scraper.scrape_and_save_all_boxscores(seasons, first_start_date)

                self.app_logger.structured_log( logging.INFO, "Boxscore scraping completed successfully")
        except Exception as e:
            # Check if it's already one of our error types (has app_logger)
            if hasattr(e, 'app_logger'):
                raise
            self.app_logger.structured_log( logging.ERROR, "Unexpected error in scrape_and_save_all_boxscores",
                           error_message=str(e),
                           error_type=type(e).__name__)
            raise self.error_handler.create_error_handler('scraping', f"Unexpected error occurred while scraping boxscores: {str(e)}")

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
                self.app_logger.structured_log( logging.INFO, "Starting to scrape matchups",
                               search_day=search_day)

                if self._schedule_scraper.scrape_and_save_matchups_for_day(search_day):
                    self.app_logger.structured_log( logging.INFO, "Matchup scraping completed successfully")
                    return True
                else:
                    self.app_logger.structured_log( logging.INFO, "No matchups found for the given day")
                    return False

        except Exception as e:
            # Check if it's already one of our error types (has app_logger)
            if hasattr(e, 'app_logger'):
                raise
            self.app_logger.structured_log( logging.ERROR, "Unexpected error in scrape_and_save_matchups_for_day",
                           error_message=str(e),
                           error_type=type(e).__name__)
            raise self.error_handler.create_error_handler('scraping', f"Unexpected error occurred while scraping matchups: {str(e)}")

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
            raise self.error_handler.create_error_handler('data_validation', "Seasons list cannot be empty")

        # Check date format MM/DD/YYYY
        if not isinstance(first_start_date, str) or not re.match(r'^\d{2}/\d{2}/\d{4}$', first_start_date):
            raise self.error_handler.create_error_handler('data_validation', "Invalid first_start_date format. Expected MM/DD/YYYY")

    def _validate_search_day(self, search_day: str) -> None:
        """
        Validate the search_day parameter for scraping matchups.

        Args:
            search_day (str): The day to search for matchups.

        Raises:
            DataValidationError: If the search_day parameter is invalid.
        """
        if not isinstance(search_day, str) or len(search_day) != 3:
            raise self.error_handler.create_error_handler('data_validation', "Invalid search_day format. Expected 3-letter day abbreviation (e.g., 'MON', 'TUE')")

  