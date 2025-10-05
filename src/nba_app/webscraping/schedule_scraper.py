"""
schedule_scraper.py

This module contains the ScheduleScraper class, which is responsible for scraping NBA schedule data.
It provides functionality to extract matchups and game IDs for specific days from the NBA website.
The class implements the BaseScheduleScraper interface and uses custom exceptions for more
specific error handling. It also includes enhanced logging for better debugging and monitoring.
"""

import logging
from typing import List, Optional
import pandas as pd
from selenium.webdriver.remote.webelement import WebElement

from .base_scraper_classes import (
    BaseScheduleScraper,
    BasePageScraper,
)
from ml_framework.framework.data_access.base_data_access import BaseDataAccess
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.error_handling.error_handler_factory import ErrorHandlerFactory
from ml_framework.core.app_logging import log_performance, log_context, structured_log, AppLogger

class ScheduleScraper(BaseScheduleScraper):
    """
    A class for scraping NBA schedule data.

    This class provides methods to scrape matchups and game IDs for specific days.

    Attributes:
        config (BaseConfigManager): Configuration object.
        data_access (BaseDataAccess): Data access object.
        page_scraper (BasePageScraper): An instance of PageScraper.
    """

    @log_performance
    def __init__(self, config: BaseConfigManager, data_access: BaseDataAccess, page_scraper: BasePageScraper, app_logger: AppLogger, error_handler: ErrorHandlerFactory) -> None:
        """
        Initialize the ScheduleScraper with configuration, data access, and page scraper.

        Args:
            config (BaseConfigManager): Configuration object.
            data_access (BaseDataAccess): Data access object.
            page_scraper (BasePageScraper): Page scraper object.
            app_logger (AppLogger): Application logger instance.
            error_handler (ErrorHandlerFactory): Error handler factory instance.
        """
        self.config = config
        self.data_access = data_access
        self.page_scraper = page_scraper
        self.app_logger = app_logger
        self.error_handler = error_handler
        self.app_logger.structured_log(logging.INFO, "ScheduleScraper initialized",
                       config_type=type(config).__name__,
                       data_access_type=type(data_access).__name__,
                       page_scraper_type=type(page_scraper).__name__)

    @log_performance
    def scrape_and_save_matchups_for_day(self, search_day: str) -> bool:
        """
        Scrape and save matchups for a specific day.

        Args:
            search_day (str): The day to search for matchups (e.g., 'MON', 'TUE').

        Returns:
            bool: True if matchups were found and saved, False otherwise.

        Raises:
            PageLoadError: If the page load times out.
            ElementNotFoundError: If required elements are not found on the page.
            ScrapingError: If there's an error during the scraping process.
            DataStorageError: If there's an error saving the scraped data.
        """
        with log_context(operation="scrape_matchups", search_day=search_day):
            try:
                self.app_logger.structured_log( logging.INFO, "Starting to scrape matchups", search_day=search_day)
                
                self.page_scraper.go_to_url(self.config.nba_schedule_url)
                self.app_logger.structured_log( logging.INFO, "Navigated to NBA schedule URL")

                days_games = self._find_games_for_day(search_day)
                
                if days_games is None:
                    # no games for the day
                    return False

                matchups = self._extract_team_ids_schedule(days_games)
                games = self._extract_game_ids_schedule(days_games)

                self._save_matchups_and_games(matchups, games)

                self.app_logger.structured_log( logging.INFO, "Successfully scraped and saved matchups", 
                               matchups_count=len(matchups), games_count=len(games))

                return True

            except Exception as e:
                # Check if it's already one of our error types (has app_logger)
                if hasattr(e, 'app_logger'):
                    raise
                self.app_logger.structured_log( logging.ERROR, "Unexpected error occurred",
                               error_message=str(e), error_type=type(e).__name__)
                raise self.error_handler.create_error_handler('scraping', f"Unexpected error during scraping: {str(e)}")

    @log_performance
    def _find_games_for_day(self, search_day: str) -> Optional[WebElement]:
        """
        Find games for a specific day on the schedule page.

        Args:
            search_day (str): The day to search for games.

        Returns:
            Optional[WebElement]: A WebElement containing the games for the day, or None if not found.

        Raises:
            ElementNotFoundError: If the required elements are not found on the page.
            ScrapingError: If there's an error during the scraping process.
        """
        try:
            self.app_logger.structured_log( logging.INFO, "Searching for games", search_day=search_day)
            
            game_days = self.page_scraper.get_elements_by_class(self.config.day_class_name)
            games_containers = self.page_scraper.get_elements_by_class(self.config.games_per_day_class_name)

            if not game_days or not games_containers:
                raise self.error_handler.create_error_handler('element_not_found', "Game days or games containers not found on the page")
            
            for day, days_games in zip(game_days, games_containers):
                if search_day.upper() == day.text[:3].upper():
                    self.app_logger.structured_log( logging.INFO, "Found games for the specified day", search_day=search_day)
                    return days_games


            self.app_logger.structured_log( logging.WARNING, "No games found for the specified day", search_day=search_day)
            return None
        except Exception as e:
            # Check if it's already one of our error types (has app_logger)
            if hasattr(e, 'app_logger'):
                raise
            self.app_logger.structured_log( logging.ERROR, "Error finding games for day",
                           search_day=search_day, error_message=str(e), error_type=type(e).__name__)
            raise self.error_handler.create_error_handler('scraping', f"Error finding games for day {search_day}: {str(e)}")

    @log_performance
    def _extract_team_ids_schedule(self, todays_games: WebElement) -> List[List[str]]:
        """
        Extract team IDs from the schedule page.

        Args:
            todays_games (WebElement): A WebElement containing the games for the day.

        Returns:
            List[List[str]]: A list of lists containing visitor and home team IDs.

        Raises:
            ElementNotFoundError: If the team links are not found on the page.
            DataExtractionError: If there's an error extracting team IDs.
        """
        try:
            self.app_logger.structured_log( logging.INFO, "Extracting team IDs from schedule")
                        
            links = self.page_scraper.get_links_by_class(self.config.teams_links_class_name, todays_games)

            if not links:
                raise self.error_handler.create_error_handler('element_not_found', "Team links not found on the page")

            teams_list = [i.get_attribute("href") for i in links]

            matchups = []
            for i in range(0, len(teams_list), 2):
                visitor_id = teams_list[i].partition("team/")[2].partition("/")[0]
                home_id = teams_list[i+1].partition("team/")[2].partition("/")[0]
                matchups.append([visitor_id, home_id])


            self.app_logger.structured_log( logging.INFO, "Team IDs extracted successfully", matchups_count=len(matchups))
            return matchups
        except Exception as e:
            # Check if it's already one of our error types (has app_logger)
            if hasattr(e, 'app_logger'):
                raise
            self.app_logger.structured_log( logging.ERROR, "Error extracting team IDs",
                           error_message=str(e), error_type=type(e).__name__)
            raise self.error_handler.create_error_handler('data_extraction', f"Error extracting team IDs: {str(e)}")

    @log_performance
    def _extract_game_ids_schedule(self, todays_games: WebElement) -> List[str]:
        """
        Extract game IDs from the schedule page.

        Args:
            todays_games (WebElement): A WebElement containing the games for the day.

        Returns:
            List[str]: A list of game IDs.

        Raises:
            ElementNotFoundError: If the game links are not found on the page.
            DataExtractionError: If there's an error extracting game IDs.
        """
        try:
            self.app_logger.structured_log( logging.INFO, "Extracting game IDs from schedule")
            
            links = self.page_scraper.get_links_by_class(self.config.game_links_class_name, todays_games)

            if not links:
                raise self.error_handler.create_error_handler('element_not_found', "Game links not found on the page")

            links = [i for i in links if self.config.schedule_preview_text in i.text]
            game_id_list = [i.get_attribute("href") for i in links]
            
            games = []
            for game in game_id_list:
                game_id = game.partition("-00")[2]
                if len(game_id) > 0:               
                    games.append(game_id)


            self.app_logger.structured_log( logging.INFO, "Game IDs extracted successfully", games_count=len(games))
            return games
        except Exception as e:
            # Check if it's already one of our error types (has app_logger)
            if hasattr(e, 'app_logger'):
                raise
            self.app_logger.structured_log( logging.ERROR, "Error extracting game IDs",
                           error_message=str(e), error_type=type(e).__name__)
            raise self.error_handler.create_error_handler('data_extraction', f"Error extracting game IDs: {str(e)}")

    @log_performance
    def _save_matchups_and_games(self, matchups: List[List[str]], games: List[str]) -> None:
        """
        Converts matchups and game IDs dataframes and saves them to CSV files.

        Args:
            matchups (List[List[str]]): A list of lists containing visitor and home team IDs.
            games (List[str]): A list of game IDs.

        Raises:
            DataValidationError: If the input data is invalid.
            DataStorageError: If there's an error while saving the data.
        """
        try:
            self.app_logger.structured_log( logging.INFO, "Saving matchups and games", 
                           matchups_count=len(matchups), games_count=len(games))
            schedule_dataframes = []
            file_names = []
            if not matchups or not games:
                raise self.error_handler.create_error_handler('data_validation', "Matchups or games list is empty")

            matchups_df = pd.DataFrame(matchups, columns=[self.config.schedule_visitor_team_id_column, self.config.schedule_home_team_id_column])
            schedule_dataframes.append(matchups_df)
            file_names.append(self.config.todays_matchups_file)

            games_df = pd.DataFrame(games, columns=[self.config.schedule_game_id_column])
            schedule_dataframes.append(games_df)
            file_names.append(self.config.todays_games_ids_file)

            self.data_access.save_dataframes(schedule_dataframes, file_names)
            self.app_logger.structured_log( logging.INFO, "Successfully saved matchups and game IDs data")

        except Exception as e:
            # Check if it's already one of our error types (has app_logger)
            if hasattr(e, 'app_logger'):
                raise
            self.app_logger.structured_log( logging.ERROR, "Error saving matchups and games data",
                           error_message=str(e), error_type=type(e).__name__)
            raise self.error_handler.create_error_handler('data_storage', f"Error saving matchups and games data: {str(e)}")