"""
schedule_scraper.py

This module contains the ScheduleScraper class, which is responsible for scraping NBA schedule data.
It provides functionality to extract matchups and game IDs for specific days from the NBA website.
The class implements the AbstractScheduleScraper interface and uses custom exceptions for more
specific error handling. It also includes enhanced logging for better debugging and monitoring.
"""

import logging
from typing import List, Optional
import pandas as pd
from selenium.webdriver.remote.webelement import WebElement

from .abstract_scraper_classes import (
    AbstractScheduleScraper, 
    AbstractPageScraper,
)
from ..data_access.abstract_data_access import AbstractDataAccess
from ..config.config import AbstractConfig
from ..error_handling.custom_exceptions import (
    ScrapingError, DataExtractionError, DataValidationError, ElementNotFoundError,
    PageLoadError, DataStorageError
)
from ..logging.logging_utils import log_performance, log_context, structured_log

logger = logging.getLogger(__name__)

class ScheduleScraper(AbstractScheduleScraper):
    """
    A class for scraping NBA schedule data.

    This class provides methods to scrape matchups and game IDs for specific days.

    Attributes:
        config (AbstractConfig): Configuration object.
        data_access (AbstractDataAccess): Data access object.
        page_scraper (AbstractPageScraper): An instance of PageScraper.
    """

    @log_performance
    def __init__(self, config: AbstractConfig, data_access: AbstractDataAccess, page_scraper: AbstractPageScraper) -> None:
        """
        Initialize the ScheduleScraper with configuration, data access, and page scraper.

        Args:
            config (AbstractConfig): Configuration object.
            data_access (AbstractDataAccess): Data access object.
            page_scraper (AbstractPageScraper): Page scraper object.
        """
        self.config = config
        self.data_access = data_access
        self.page_scraper = page_scraper
        structured_log(logger, logging.INFO, "ScheduleScraper initialized", 
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
                structured_log(logger, logging.INFO, "Starting to scrape matchups", search_day=search_day)
                
                self.page_scraper.go_to_url(self.config.nba_schedule_url)
                structured_log(logger, logging.INFO, "Navigated to NBA schedule URL")

                days_games = self._find_games_for_day(search_day)
                
                if days_games is None:
                    # no games for the day
                    return False

                matchups = self._extract_team_ids_schedule(days_games)
                games = self._extract_game_ids_schedule(days_games)

                self._save_matchups_and_games(matchups, games)

                structured_log(logger, logging.INFO, "Successfully scraped and saved matchups", 
                               matchups_count=len(matchups), games_count=len(games))

                return True

            except (PageLoadError, ElementNotFoundError, ScrapingError, DataStorageError) as e:
                structured_log(logger, logging.ERROR, "Error during scraping process", 
                               error_message=str(e), error_type=type(e).__name__)
                raise
            except Exception as e:
                structured_log(logger, logging.ERROR, "Unexpected error occurred", 
                               error_message=str(e), error_type=type(e).__name__)
                raise ScrapingError(f"Unexpected error during scraping: {str(e)}")

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
            structured_log(logger, logging.INFO, "Searching for games", search_day=search_day)
            
            game_days = self.page_scraper.get_elements_by_class(self.config.day_class_name)
            games_containers = self.page_scraper.get_elements_by_class(self.config.games_per_day_class_name)

            if not game_days or not games_containers:
                raise ElementNotFoundError("Game days or games containers not found on the page")
            
            for day, days_games in zip(game_days, games_containers):
                if search_day == day.text[:3]:
                    structured_log(logger, logging.INFO, "Found games for the specified day", search_day=search_day)
                    return days_games
            
            structured_log(logger, logging.WARNING, "No games found for the specified day", search_day=search_day)
            return None
        except ElementNotFoundError:
            raise
        except Exception as e:
            structured_log(logger, logging.ERROR, "Error finding games for day", 
                           search_day=search_day, error_message=str(e), error_type=type(e).__name__)
            raise ScrapingError(f"Error finding games for day {search_day}: {str(e)}")

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
            structured_log(logger, logging.INFO, "Extracting team IDs from schedule")
            
            links = self.page_scraper.get_elements_by_class(self.config.teams_links_class_name, todays_games)
            
            if not links:
                raise ElementNotFoundError("Team links not found on the page")

            teams_list = [i.get_attribute("href") for i in links]

            matchups = []
            for i in range(0, len(teams_list), 2):
                visitor_id = teams_list[i].partition("team/")[2].partition("/")[0]
                home_id = teams_list[i+1].partition("team/")[2].partition("/")[0]
                matchups.append([visitor_id, home_id])
            
            structured_log(logger, logging.INFO, "Team IDs extracted successfully", matchups_count=len(matchups))
            return matchups
        except ElementNotFoundError:
            raise
        except Exception as e:
            structured_log(logger, logging.ERROR, "Error extracting team IDs", 
                           error_message=str(e), error_type=type(e).__name__)
            raise DataExtractionError(f"Error extracting team IDs: {str(e)}")

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
            structured_log(logger, logging.INFO, "Extracting game IDs from schedule")
            
            links = self.page_scraper.get_elements_by_class(self.config.game_links_class_name, todays_games)
            
            if not links:
                raise ElementNotFoundError("Game links not found on the page")

            links = [i for i in links if "PREVIEW" in i.text]
            game_id_list = [i.get_attribute("href") for i in links]
            
            games = []
            for game in game_id_list:
                game_id = game.partition("-00")[2].partition("?")[0]
                if len(game_id) > 0:               
                    games.append(game_id)
            
            structured_log(logger, logging.INFO, "Game IDs extracted successfully", games_count=len(games))
            return games
        except ElementNotFoundError:
            raise
        except Exception as e:
            structured_log(logger, logging.ERROR, "Error extracting game IDs", 
                           error_message=str(e), error_type=type(e).__name__)
            raise DataExtractionError(f"Error extracting game IDs: {str(e)}")

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
            structured_log(logger, logging.INFO, "Saving matchups and games", 
                           matchups_count=len(matchups), games_count=len(games))
            schedule_dataframes = []
            file_names = []
            if not matchups or not games:
                raise DataValidationError("Matchups or games list is empty")

            matchups_df = pd.DataFrame(matchups, columns=['visitor_id', 'home_id'])
            schedule_dataframes.append(matchups_df)
            file_names.append(self.config.todays_matchups_file)

            games_df = pd.DataFrame(games, columns=['game_id'])
            schedule_dataframes.append(games_df)
            file_names.append(self.config.todays_games_ids_file)

            self.data_access.save_dataframes(schedule_dataframes, file_names)
            structured_log(logger, logging.INFO, "Successfully saved matchups and game IDs data")
            
        except DataValidationError:
            raise
        except Exception as e:
            structured_log(logger, logging.ERROR, "Error saving matchups and games data", 
                           error_message=str(e), error_type=type(e).__name__)
            raise DataStorageError(f"Error saving matchups and games data: {str(e)}")