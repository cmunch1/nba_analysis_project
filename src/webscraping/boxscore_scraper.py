"""
boxscore_scraper.py

This module contains the BoxscoreScraper class, which is responsible for scraping NBA boxscore data
from the official NBA stats website. It implements the AbstractBoxscoreScraper interface.
"""

import logging
from typing import List, Optional, Tuple
from datetime import datetime
import pandas as pd
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.remote.webdriver import WebDriver

from .abstract_scraper_classes import AbstractBoxscoreScraper, AbstractPageScraper
from .page_scraper import PageScraper
from ..config.config import config
from ..data_access.data_access import DataAccess

data_access = DataAccess()

class BoxscoreScraper(AbstractBoxscoreScraper):
    """
    A class for scraping NBA boxscore data.

    This class provides methods to scrape boxscores for multiple seasons and stat types.

    Attributes:
        driver (WebDriver): A Selenium WebDriver instance.
        page_scraper (AbstractPageScraper): An instance of PageScraper.
        logger (logging.Logger): Logger for this class.
    """

    def __init__(self, driver: WebDriver):
        """
        Initialize the BoxscoreScraper with a WebDriver and load configuration.

        Args:
            driver (WebDriver): A Selenium WebDriver instance.
        """
        self.driver = driver
        self.page_scraper: AbstractPageScraper = PageScraper(driver)
        self.logger = logging.getLogger(__name__)

    def scrape_and_save_all_boxscores(self, seasons: List[str], first_start_date: str) -> None:
        """
        Scrape and save boxscores for all stat types and specified seasons.

        Args:
            seasons (List[str]): A list of seasons to scrape.
            first_start_date (str): The start date for the first season.

        Raises:
            ValueError: If seasons list is empty or first_start_date is invalid.
        """
        if not seasons:
            raise ValueError("Seasons list cannot be empty")
        if not self._is_valid_date(first_start_date):
            raise ValueError(f"Invalid first_start_date: {first_start_date}")

        for stat_type in config.stat_types:
            try:
                new_games = self.scrape_stat_type(seasons, first_start_date, stat_type)
                file_name = f"games_{stat_type}.csv"
                data_access.save_scraped_data(new_games, file_name)
                self.logger.info(f"Successfully scraped and saved {stat_type} stats for {len(seasons)} seasons")
            except Exception as e:
                self.logger.error(f"Error scraping {stat_type} stats: {str(e)}")

    def scrape_stat_type(self, seasons: List[str], first_start_date: str, stat_type: str) -> pd.DataFrame:
        """
        Scrape a specific stat type for multiple seasons.

        Args:
            seasons (List[str]): A list of seasons to scrape.
            first_start_date (str): The start date for the first season.
            stat_type (str): The type of stats to scrape.

        Returns:
            pd.DataFrame: A DataFrame with scraped data for all seasons.

        Raises:
            ValueError: If stat_type is not supported.
        """
        if stat_type not in config.stat_types:
            raise ValueError(f"Unsupported stat type: {stat_type}")

        new_games = pd.DataFrame()
        start_date = first_start_date

        for season in seasons:
            season_year = int(season[:4])    
            end_date = f"{config.off_season_start_month}/01/{season_year+1}"
            df_season = self.scrape_sub_seasons(str(season), str(start_date), str(end_date), stat_type)
            new_games = pd.concat([new_games, df_season], axis=0)
            start_date = f"{config.regular_season_start_month}/01/{season_year+1}"

        return new_games

    def scrape_sub_seasons(self, season: str, start_date: str, end_date: str, stat_type: str) -> pd.DataFrame:
        """
        Scrape data for all sub-seasons within a given season.

        Args:
            season (str): The season to scrape.
            start_date (str): The start date of the season.
            end_date (str): The end date of the season.
            stat_type (str): The type of stats to scrape.

        Returns:
            pd.DataFrame: A DataFrame containing scraped data for all sub-seasons.
        """
        self.logger.info(f"Scraping {season} from {start_date} to {end_date} for {stat_type} stats")
        
        all_sub_seasons = pd.DataFrame()
        sub_season_types = self.determine_sub_season_types(start_date, end_date)

        for sub_season_type in sub_season_types:
            try:
                df = self.scrape_to_dataframe(Season=season, DateFrom=start_date, DateTo=end_date, stat_type=stat_type, season_type=sub_season_type)
                if not df.empty:
                    all_sub_seasons = pd.concat([all_sub_seasons, df], axis=0)
            except Exception as e:
                self.logger.error(f"Error scraping {sub_season_type} for {season}: {str(e)}")

        return all_sub_seasons
    
    def determine_sub_season_types(self, start_date: str, end_date: str) -> List[str]:
        """
        Determine sub-season types based on start and end dates.

        Args:
            start_date (str): The start date of the season.
            end_date (str): The end date of the season.

        Returns:
            List[str]: A list of sub-season types.
        """
        sub_season_types = []

        start_date = datetime.strptime(start_date, "%m/%d/%Y")
        end_date = datetime.strptime(end_date, "%m/%d/%Y")
        play_in_date = datetime(start_date.year, config.play_in_month, 1)

        if start_date < play_in_date and end_date < play_in_date:
            sub_season_types.append(config.regular_season_text)
        elif start_date > play_in_date and end_date > play_in_date:
            sub_season_types.append(config.playoffs_season_text)
        else:
            sub_season_types.append(config.regular_season_text)
            sub_season_types.append(config.playoffs_season_text)
            sub_season_types.append(config.play_in_season_text)

        return sub_season_types

    def scrape_to_dataframe(self, Season: str, DateFrom: str = "NONE", DateTo: str = "NONE", stat_type: str = 'traditional', season_type: str = "Regular+Season") -> pd.DataFrame:
        """
        Scrape data and convert it to a DataFrame.

        Args:
            Season (str): The season to scrape.
            DateFrom (str, optional): The start date. Defaults to "NONE".
            DateTo (str, optional): The end date. Defaults to "NONE".
            stat_type (str, optional): The type of stats to scrape. Defaults to 'traditional'.
            season_type (str, optional): The type of season. Defaults to "Regular+Season".

        Returns:
            pd.DataFrame: A DataFrame containing scraped data.
        """
        data_table = self.scrape_boxscores_table(Season, DateFrom, DateTo, stat_type, season_type)
        
        if data_table is None:
            self.logger.warning(f"No data found for {Season} {season_type} {stat_type}")
            return pd.DataFrame()
        
        return self.convert_table_to_df(data_table)

    def scrape_boxscores_table(self, Season: str, DateFrom: str = "NONE", DateTo: str = "NONE", stat_type: str = 'traditional', season_type: str = "Regular+Season") -> Optional[WebElement]:
        """
        Scrape the boxscores table from the NBA stats website.

        Args:
            Season (str): The season to scrape.
            DateFrom (str, optional): The start date. Defaults to "NONE".
            DateTo (str, optional): The end date. Defaults to "NONE".
            stat_type (str, optional): The type of stats to scrape. Defaults to 'traditional'.
            season_type (str, optional): The type of season. Defaults to "Regular+Season".

        Returns:
            Optional[WebElement]: A WebElement containing the scraped table, or None if not found.
        """
        nba_url = self._construct_nba_url(stat_type, season_type, Season, DateFrom, DateTo)
        self.logger.info(f"Scraping {nba_url}")

        return self.page_scraper.scrape_page_table(nba_url, config.table_class_name, config.pagination_class_name, config.dropdown_class_name)

    def convert_table_to_df(self, data_table: WebElement) -> pd.DataFrame:
        """
        Convert a WebElement table to a DataFrame.

        Args:
            data_table (WebElement): A WebElement containing the table data.

        Returns:
            pd.DataFrame: A DataFrame representation of the table.
        """
        table_html = data_table.get_attribute('outerHTML')
        dfs = pd.read_html(table_html, header=0)
        df = pd.concat(dfs)

        team_id, game_id = self._extract_team_and_game_ids_boxscores(data_table)
        df['TEAM_ID'] = team_id
        df['GAME_ID'] = game_id

        return df

    def _construct_nba_url(self, stat_type: str, season_type: str, Season: str, DateFrom: str, DateTo: str) -> str:
        """
        Construct the URL for NBA stats website based on given parameters.

        Args:
            stat_type (str): The type of stats to scrape.
            season_type (str): The type of season.
            Season (str): The season to scrape.
            DateFrom (str): The start date.
            DateTo (str): The end date.

        Returns:
            str: The constructed URL string.
        """
        base_url = f"{config.nba_boxscores_url}-{stat_type}" if stat_type != 'traditional' else config.nba_boxscores_url
        nba_url = f"{base_url}?SeasonType={season_type}"
        
        if not Season:
            nba_url = f"{nba_url}&DateFrom={DateFrom}&DateTo={DateTo}"
        else:
            if DateFrom == "NONE" and DateTo == "NONE":
                nba_url = f"{nba_url}&Season={Season}"
            else:
                nba_url = f"{nba_url}&Season={Season}&DateFrom={DateFrom}&DateTo={DateTo}"
        return nba_url

    def _extract_team_and_game_ids_boxscores(self, data_table: WebElement) -> Tuple[pd.Series, pd.Series]:
        """
        Extract team and game IDs from the boxscores table.

        Args:
            data_table (WebElement): A WebElement containing the table data.

        Returns:
            Tuple[pd.Series, pd.Series]: A tuple of Series containing team IDs and game IDs.
        """
        links = self.page_scraper.get_elements_by_class(config.teams_and_games_class_name, data_table)       
        links_list = [i.get_attribute("href") for i in links]
    
        team_id = pd.Series([i[-10:] for i in links_list if ('stats' in i)])
        game_id = pd.Series([i[-10:] for i in links_list if ('/game/' in i)])
    
        return team_id, game_id

    @staticmethod
    def _is_valid_date(date_string: str) -> bool:
        """
        Check if a given date string is valid.

        Args:
            date_string (str): The date string to validate.

        Returns:
            bool: True if the date is valid, False otherwise.
        """
        try:
            datetime.strptime(date_string, "%m/%d/%Y")
            return True
        except ValueError:
            return False
