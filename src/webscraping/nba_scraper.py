import logging
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from selenium.webdriver.remote.webelement import WebElement

from .config import config
from .page_scraper import PageScraper
from ..data_access.data_access import save_scraped_data


class NbaScraper:
    """
    A class for scraping NBA data from official NBA stats website.

    This class provides methods to scrape boxscores and game schedules,
    utilizing Selenium for web interaction and Pandas for data manipulation.
    """

    def __init__(self, driver):
        """
        Initialize the NbaScraper with a WebDriver and configuration settings.

        Args:
            driver: Selenium WebDriver instance
        """
        self.driver = driver
        self.page_scraper = PageScraper(driver)
        self._load_config()

    def __enter__(self):
        """
        Enter the runtime context related to this object.

        Returns:
            The NbaScraper instance.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the runtime context related to this object.

        Args:
            exc_type: The exception type if an exception was raised in the context.
            exc_value: The exception value if an exception was raised in the context.
            traceback: The traceback if an exception was raised in the context.

        Returns:
            None
        """

        self.driver.quit()
        
        # If you want to handle exceptions, you can do so here
        if exc_type is not None:
            self.logger.error(f"An error occurred: {exc_type}, {exc_value}")
        
        # Returning False (or None) will propagate exceptions
        return False


    def _load_config(self) -> None:
        """Load configuration settings from config file."""
        self.start_season = config["start_season"]
        self.regular_season_start_month = config["regular_season_start_month"]
        self.off_season_start_month = config["off_season_start_month"]
        # boxscores sub-pages - url construction requires that these be specified
        self.sub_season_types = config["sub_season_types"] #["Regular+Season", "PlayIn", "Playoffs"],
        self.stat_types = config["stat_types"] #["traditional", "advanced", "four-factors", "misc", "scoring"],
        # boxscores
        self.nba_boxscores_url = config["nba_boxscores_url"]
        self.table_class_name = config["table_class_name"]
        self.pagination_class_name = config["pagination_class_name"]
        self.dropdown_class_name = config["dropdown_class_name"]
        self.teams_and_games_class_name = config["teams_and_games_class_name"]
        # schedule
        self.nba_schedule_url = config["nba_schedule_url"]
        self.games_per_day_class_name = config["games_per_day_class_name"]
        self.day_class_name = config["day_class_name"]
        self.teams_links_class_name = config["teams_links_class_name"]
        self.game_links_class_name = config["game_links_class_name"]
        # other
        self.max_retries = config["max_retries"]
        self.retry_delay = config["retry_delay"]
        self.wait_time = config["wait_time"]
        self.log_level = config["log_level"]

        # Set up logging based on config
        logging.basicConfig(level=getattr(logging, self.log_level),
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)

    def scrape_and_save_all_boxscores(self, seasons: List[str], first_start_date: str) -> None:
        """
        Scrape and save boxscores for all stat types and specified seasons.

        Args:
            seasons: List of seasons to scrape
            first_start_date: Start date for the first season
        """
        for stat_type in self.stat_types:
            new_games = self.scrape_stat_type(seasons, first_start_date, stat_type)
            file_name = f"games_{stat_type}.csv"
            save_scraped_data(new_games, file_name)

    def scrape_stat_type(self, seasons: List[str], first_start_date: str, stat_type: str) -> Tuple[str, pd.DataFrame]:
        """
        Scrape a specific stat type for multiple seasons.

        Args:
            seasons: List of seasons to scrape
            first_start_date: Start date for the first season
            stat_type: Type of stats to scrape

        Returns:
            Tuple containing the stat type and DataFrame with scraped data for all seasons
        """
        new_games = pd.DataFrame()
        start_date = first_start_date

        for season in seasons:
            season_year = int(season[:4])    
            end_date = f"{self.off_season_start_month}/01/{season_year+1}"
            df_season = self.scrape_sub_seasons(str(season), str(start_date), str(end_date), stat_type)
            new_games = pd.concat([new_games, df_season], axis=0)
            start_date = f"{self.regular_season_start_month}/01/{season_year+1}"

        return stat_type, new_games

    def scrape_sub_seasons(self, season: str, start_date: str, end_date: str, stat_type: str) -> pd.DataFrame:
        """
        Scrape data for all sub-seasons within a given season.

        Args:
            season: Season to scrape
            start_date: Start date of the season
            end_date: End date of the season
            stat_type: Type of stats to scrape

        Returns:
            DataFrame containing scraped data for all sub-seasons
        """
        self.logger.info(f"Scraping {season} from {start_date} to {end_date} for {stat_type} stats")
        
        all_sub_seasons = pd.DataFrame()

        for sub_season_type in self.sub_season_types:
            df = self.scrape_to_dataframe(Season=season, DateFrom=start_date, DateTo=end_date, stat_type=stat_type, season_type=sub_season_type)
            if not df.empty:
                all_sub_seasons = pd.concat([all_sub_seasons, df], axis=0)

        return all_sub_seasons
    
    def scrape_to_dataframe(self, Season: str, DateFrom: str = "NONE", DateTo: str = "NONE", stat_type: str = 'traditional', season_type: str = "Regular+Season") -> pd.DataFrame:
        """
        Scrape data and convert it to a DataFrame.

        Args:
            Season: Season to scrape
            DateFrom: Start date (default "NONE")
            DateTo: End date (default "NONE")
            stat_type: Type of stats to scrape (default 'traditional')
            season_type: Type of season (default "Regular+Season")

        Returns:
            DataFrame containing scraped data
        """
        data_table = self.scrape_boxscores_table(Season, DateFrom, DateTo, stat_type, season_type)
        
        if data_table is None:
            return pd.DataFrame()
        
        return self.convert_table_to_df(data_table)

    def scrape_boxscores_table(self, Season: str, DateFrom: str = "NONE", DateTo: str = "NONE", stat_type: str = 'traditional', season_type: str = "Regular+Season") -> Optional[WebElement]:
        """
        Scrape the boxscores table from the NBA stats website.

        Args:
            Season: Season to scrape
            DateFrom: Start date (default "NONE")
            DateTo: End date (default "NONE")
            stat_type: Type of stats to scrape (default 'traditional')
            season_type: Type of season (default "Regular+Season")

        Returns:
            WebElement containing the scraped table, or None if not found
        """
        nba_url = self._construct_nba_url(stat_type, season_type, Season, DateFrom, DateTo)
        self.logger.info(f"Scraping {nba_url}")

        return self.page_scraper.scrape_page_table(nba_url, self.table_class_name, self.pagination_class_name, self.dropdown_class_name)
    

    def convert_table_to_df(self, data_table: WebElement) -> pd.DataFrame:
        """
        Convert a WebElement table to a DataFrame.

        Args:
            data_table: WebElement containing the table data

        Returns:
            DataFrame representation of the table
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
            stat_type: Type of stats to scrape
            season_type: Type of season
            Season: Season to scrape
            DateFrom: Start date
            DateTo: End date

        Returns:
            Constructed URL string
        """
        base_url = f"{self.nba_boxscores_url}-{stat_type}" if stat_type != 'traditional' else self.nba_boxscores_url
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
            data_table: WebElement containing the table data

        Returns:
            Tuple of Series containing team IDs and game IDs
        """
        links = self.page_scraper.get_elements_by_class(self.teams_and_games_class_name, data_table)       
        links_list = [i.get_attribute("href") for i in links]
    
        team_id = pd.Series([i[-10:] for i in links_list if ('stats' in i)])
        game_id = pd.Series([i[-10:] for i in links_list if ('/game/' in i)])
    
        return team_id, game_id
    
    def scrape_and_save_matchups_for_day(self, search_day: str) -> None:
        """
        Scrape and save matchups for a specific day.

        Args:
            search_day: Day to search for matchups
        """
        self.page_scraper.go_to_url(self.nba_schedule_url)

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
            search_day: Day to search for games

        Returns:
            WebElement containing the games for the day, or None if not found
        """
        game_days = self.page_scraper.get_elements_by_class(self.day_class_name)
        games_containers = self.page_scraper.get_elements_by_class(self.games_per_day_class_name)
        
        for day, days_games in zip(game_days, games_containers):
            if search_day == day.text[:3]:
                return days_games
        return None

    def _extract_team_ids_schedule(self, todays_games: WebElement) -> List[List[str]]:
        """
        Extract team IDs from the schedule page.

        Args:
            todays_games: WebElement containing the games for the day

        Returns:
            List of lists containing visitor and home team IDs
        """
        links = self.page_scraper.get_elements_by_class(self.teams_links_class_name, todays_games)
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
            todays_games: WebElement containing the games for the day

        Returns:
            List of game IDs
        """
        links = self.page_scraper.get_elements_by_class(self.game_links_class_name, todays_games)
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
            matchups: List of lists containing visitor and home team IDs
            games: List of game IDs
        """
        try:
            matchups_df = pd.DataFrame(matchups)
            save_scraped_data(matchups_df, "matchups")
            self.logger.info("Successfully saved matchups data")

            games_df = pd.DataFrame(games)
            save_scraped_data(games_df, "games_ids")
            self.logger.info("Successfully saved game IDs data")
        except Exception as e:
            self.logger.error(f"Error saving matchups and games data: {str(e)}")






