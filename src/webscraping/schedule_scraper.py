"""
schedule_scraper.py

This module contains the ScheduleScraper class, which is responsible for scraping NBA schedule data.
It provides functionality to extract matchups and game IDs for specific days from the NBA website.
The class implements the AbstractScheduleScraper interface.

Key features:
- Scrapes matchups and game IDs for a given day
- Saves scraped data to CSV files
- Implements error handling and logging
- Uses Selenium WebDriver for web scraping
"""

import logging
from typing import List, Optional
import pandas as pd
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException

from .abstract_scraper_classes import (
    AbstractScheduleScraper, 
    AbstractPageScraper,
)
from ..data_access.abstract_data_access import (
    AbstractDataAccess
)
from ..config.config import AbstractConfig



class ScheduleScraper(AbstractScheduleScraper):
    """
    A class for scraping NBA schedule data.

    This class provides methods to scrape matchups and game IDs for specific days.

    Attributes:
        
        page_scraper (AbstractPageScraper): An instance of PageScraper.
        logger (logging.Logger): A logging instance for this class.
    """

    def __init__(self, config: AbstractConfig, data_access: AbstractDataAccess, page_scraper: AbstractPageScraper) -> None:
        """
        Initialize the ScheduleScraper with a WebDriver and load configuration.

        Args:
            
        """
        self.config = config
        self.data_access = data_access
        self.page_scraper = page_scraper
        self.logger = logging.getLogger(__name__)

    def scrape_and_save_matchups_for_day(self, search_day: str) -> None:
        """
        Scrape and save matchups for a specific day.

        Args:
            search_day (str): The day to search for matchups (e.g., 'MON', 'TUE').

        Raises:
            TimeoutException: If the page load times out.
            NoSuchElementException: If required elements are not found on the page.
        """
        try:
            self.page_scraper.go_to_url(self.config.nba_schedule_url)

            days_games = self._find_games_for_day(search_day)
            
            if days_games is None:
                self.logger.warning(f"No games found for {search_day}")
                return

            matchups = self._extract_team_ids_schedule(days_games)
            games = self._extract_game_ids_schedule(days_games)

            self._save_matchups_and_games(matchups, games)

        except TimeoutException:
            self.logger.error(f"Timeout while loading the NBA schedule page for {search_day}")
        except NoSuchElementException as e:
            self.logger.error(f"Required element not found on the page: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error occurred: {str(e)}")

    def _find_games_for_day(self, search_day: str) -> Optional[WebElement]:
        """
        Find games for a specific day on the schedule page.

        Args:
            search_day (str): The day to search for games.

        Returns:
            Optional[WebElement]: A WebElement containing the games for the day, or None if not found.

        Raises:
            NoSuchElementException: If the required elements are not found on the page.
        """
        game_days = self.page_scraper.get_elements_by_class(self.config.day_class_name)
        games_containers = self.page_scraper.get_elements_by_class(self.config.games_per_day_class_name)

        if not game_days or not games_containers:
            raise NoSuchElementException("Game days or games containers not found on the page")
        
        for day, days_games in zip(game_days, games_containers):
            if search_day == day.text[:3]:
                return days_games
        return None

    def _extract_team_ids_schedule(self, todays_games: WebElement) -> List[List[str]]:
        """
        Extract team IDs from the schedule page.

        Args:
            todays_games (WebElement): A WebElement containing the games for the day.

        Returns:
            List[List[str]]: A list of lists containing visitor and home team IDs.

        Raises:
            NoSuchElementException: If the team links are not found on the page.
        """
        links = self.page_scraper.get_elements_by_class(self.config.teams_links_class_name, todays_games)
        
        if not links:
            raise NoSuchElementException("Team links not found on the page")

        teams_list = [i.get_attribute("href") for i in links]

        matchups = []
        for i in range(0, len(teams_list), 2):
            visitor_id = teams_list[i].partition("team/")[2].partition("/")[0]
            home_id = teams_list[i+1].partition("team/")[2].partition("/")[0]
            matchups.append([visitor_id, home_id])
        return matchups

    def _extract_game_ids_schedule(self, todays_games: WebElement) -> List[str]:
        """
        Extract game IDs from the schedule page.

        Args:
            todays_games (WebElement): A WebElement containing the games for the day.

        Returns:
            List[str]: A list of game IDs.

        Raises:
            NoSuchElementException: If the game links are not found on the page.
        """
        links = self.page_scraper.get_elements_by_class(self.config.game_links_class_name, todays_games)
        
        if not links:
            raise NoSuchElementException("Game links not found on the page")

        links = [i for i in links if "PREVIEW" in i.text]
        game_id_list = [i.get_attribute("href") for i in links]
        
        games = []
        for game in game_id_list:
            game_id = game.partition("-00")[2].partition("?")[0]
            if len(game_id) > 0:               
                games.append(game_id)
        return games

    def _save_matchups_and_games(self, matchups: List[List[str]], games: List[str]) -> None:
        """
        Save matchups and game IDs to CSV files.

        Args:
            matchups (List[List[str]]): A list of lists containing visitor and home team IDs.
            games (List[str]): A list of game IDs.

        Raises:
            Exception: If there's an error while saving the data.
        """
        try:
            matchups_df = pd.DataFrame(matchups, columns=['visitor_id', 'home_id'])
            data_access.save_scraped_data(matchups_df, "matchups")
            self.logger.info("Successfully saved matchups data")

            games_df = pd.DataFrame(games, columns=['game_id'])
            data_access.save_scraped_data(games_df, "games_ids")
            self.logger.info("Successfully saved game IDs data")
        except Exception as e:
            self.logger.error(f"Error saving matchups and games data: {str(e)}")
            raise
