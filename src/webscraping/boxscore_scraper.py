import logging
from typing import List, Optional, Tuple
import pandas as pd
from selenium.webdriver.remote.webelement import WebElement

from .config import config
from .page_scraper import PageScraper
from ..data_access.data_access import save_scraped_data


class BoxscoreScraper:
    """
    A class for scraping NBA boxscore data.

    This class provides methods to scrape boxscores for multiple seasons and stat types.

    Attributes:
        driver: A Selenium WebDriver instance.
        page_scraper: An instance of PageScraper.
        logger: A logging instance for this class.
    """

    def __init__(self, driver):
        """
        Initialize the BoxscoreScraper with a WebDriver and load configuration.

        Args:
            driver: A Selenium WebDriver instance.
        """
        self.driver = driver
        self.page_scraper = PageScraper(driver)

        logging.basicConfig(level=getattr(logging, config.log_level),
                    format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)

    def scrape_and_save_all_boxscores(self, seasons: List[str], first_start_date: str) -> None:
        """
        Scrape and save boxscores for all stat types and specified seasons.

        Args:
            seasons: A list of seasons to scrape.
            first_start_date: The start date for the first season.
        """
        for stat_type in config.stat_types:
            new_games = self.scrape_stat_type(seasons, first_start_date, stat_type)
            file_name = f"games_{stat_type}.csv"
            save_scraped_data(new_games, file_name)

    def scrape_stat_type(self, seasons: List[str], first_start_date: str, stat_type: str) -> pd.DataFrame:
        """
        Scrape a specific stat type for multiple seasons.

        Args:
            seasons: A list of seasons to scrape.
            first_start_date: The start date for the first season.
            stat_type: The type of stats to scrape.

        Returns:
            A tuple containing the stat type and a DataFrame with scraped data for all seasons.
        """
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
            season: The season to scrape.
            start_date: The start date of the season.
            end_date: The end date of the season.
            stat_type: The type of stats to scrape.

        Returns:
            A DataFrame containing scraped data for all sub-seasons.
        """
        self.logger.info(f"Scraping {season} from {start_date} to {end_date} for {stat_type} stats")
        
        all_sub_seasons = pd.DataFrame()

        for sub_season_type in config.sub_season_types:
            df = self.scrape_to_dataframe(Season=season, DateFrom=start_date, DateTo=end_date, stat_type=stat_type, season_type=sub_season_type)
            if not df.empty:
                all_sub_seasons = pd.concat([all_sub_seasons, df], axis=0)

        return all_sub_seasons
    
    def scrape_to_dataframe(self, Season: str, DateFrom: str = "NONE", DateTo: str = "NONE", stat_type: str = 'traditional', season_type: str = "Regular+Season") -> pd.DataFrame:
        """
        Scrape data and convert it to a DataFrame.

        Args:
            Season: The season to scrape.
            DateFrom: The start date (default "NONE").
            DateTo: The end date (default "NONE").
            stat_type: The type of stats to scrape (default 'traditional').
            season_type: The type of season (default "Regular+Season").

        Returns:
            A DataFrame containing scraped data.
        """
        data_table = self.scrape_boxscores_table(Season, DateFrom, DateTo, stat_type, season_type)
        
        if data_table is None:
            return pd.DataFrame()
        
        return self.convert_table_to_df(data_table)

    def scrape_boxscores_table(self, Season: str, DateFrom: str = "NONE", DateTo: str = "NONE", stat_type: str = 'traditional', season_type: str = "Regular+Season") -> Optional[WebElement]:
        """
        Scrape the boxscores table from the NBA stats website.

        Args:
            Season: The season to scrape.
            DateFrom: The start date (default "NONE").
            DateTo: The end date (default "NONE").
            stat_type: The type of stats to scrape (default 'traditional').
            season_type: The type of season (default "Regular+Season").

        Returns:
            A WebElement containing the scraped table, or None if not found.
        """
        nba_url = self._construct_nba_url(stat_type, season_type, Season, DateFrom, DateTo)
        self.logger.info(f"Scraping {nba_url}")

        return self.page_scraper.scrape_page_table(nba_url, config.table_class_name, config.pagination_class_name, config.dropdown_class_name)

    def convert_table_to_df(self, data_table: WebElement) -> pd.DataFrame:
        """
        Convert a WebElement table to a DataFrame.

        Args:
            data_table: A WebElement containing the table data.

        Returns:
            A DataFrame representation of the table.
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
            stat_type: The type of stats to scrape.
            season_type: The type of season.
            Season: The season to scrape.
            DateFrom: The start date.
            DateTo: The end date.

        Returns:
            The constructed URL string.
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
            data_table: A WebElement containing the table data.

        Returns:
            A tuple of Series containing team IDs and game IDs.
        """
        links = self.page_scraper.get_elements_by_class(config.teams_and_games_class_name, data_table)       
        links_list = [i.get_attribute("href") for i in links]
    
        team_id = pd.Series([i[-10:] for i in links_list if ('stats' in i)])
        game_id = pd.Series([i[-10:] for i in links_list if ('/game/' in i)])
    
        return team_id, game_id
