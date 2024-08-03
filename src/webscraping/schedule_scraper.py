import logging
from typing import List, Optional
import pandas as pd
from selenium.webdriver.remote.webelement import WebElement


from .page_scraper import PageScraper
from ..config.config import config
from ..data_access.data_access import DataAccess

data_access = DataAccess()

class ScheduleScraper:
    """
    A class for scraping NBA schedule data.

    This class provides methods to scrape matchups and game IDs for specific days.

    Attributes:
        driver: A Selenium WebDriver instance.
        page_scraper: An instance of PageScraper.
        logger: A logging instance for this class.
    """

    def __init__(self, driver):
        """
        Initialize the ScheduleScraper with a WebDriver and load configuration.

        Args:
            driver: A Selenium WebDriver instance.
        """
        self.driver = driver
        self.page_scraper = PageScraper(driver)
        
        logging.basicConfig(level=getattr(logging, config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)


    def scrape_and_save_matchups_for_day(self, search_day: str) -> None:
        """
        Scrape and save matchups for a specific day.

        Args:
            search_day: The day to search for matchups.
        """
        self.page_scraper.go_to_url(config.nba_schedule_url)

        days_games = self._find_games_for_day(search_day)
        
        if days_games is None:
            self.logger.warning("No games found for this day")
            return

        matchups = self._extract_team_ids_schedule(days_games)
        games = self._extract_game_ids_schedule(days_games)

        self._save_matchups_and_games(matchups, games)

    def _find_games_for_day(self, search_day: str) -> Optional[WebElement]:
        """
        Find games for a specific day on the schedule page.

        Args:
            search_day: The day to search for games.

        Returns:
            A WebElement containing the games for the day, or None if not found.
        """
        game_days = self.page_scraper.get_elements_by_class(config.day_class_name)
        games_containers = self.page_scraper.get_elements_by_class(config.games_per_day_class_name)

        if not game_days or not games_containers:
            return None
        
        for day, days_games in zip(game_days, games_containers):
            if search_day == day.text[:3]:
                return days_games
        return None

    def _extract_team_ids_schedule(self, todays_games: WebElement) -> List[List[str]]:
        """
        Extract team IDs from the schedule page.

        Args:
            todays_games: A WebElement containing the games for the day.

        Returns:
            A list of lists containing visitor and home team IDs.
        """
        links = self.page_scraper.get_elements_by_class(Config.teams_links_class_name, todays_games)
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
            todays_games: A WebElement containing the games for the day.

        Returns:
            A list of game IDs.
        """
        links = self.page_scraper.get_elements_by_class(Config.game_links_class_name, todays_games)
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
            matchups: A list of lists containing visitor and home team IDs.
            games: A list of game IDs.
        """
        try:
            matchups_df = pd.DataFrame(matchups)
            data_access.save_scraped_data(matchups_df, "matchups")
            self.logger.info("Successfully saved matchups data")

            games_df = pd.DataFrame(games)
            data_access.save_scraped_data(games_df, "games_ids")
            self.logger.info("Successfully saved game IDs data")
        except Exception as e:
            self.logger.error(f"Error saving matchups and games data: {str(e)}")
