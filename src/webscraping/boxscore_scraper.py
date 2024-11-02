"""
boxscore_scraper.py

This module contains the BoxscoreScraper class, which is responsible for scraping NBA boxscore data
from the official NBA stats website. It implements the AbstractBoxscoreScraper interface and uses
custom exceptions for more specific error handling. The module now includes enhanced logging for 
better debugging and monitoring.

Key features:
- Scrapes boxscores for multiple seasons and stat types
- Implements comprehensive logging using structured logging
- Uses context managers for consistent log formatting
- Implements granular error handling with custom exceptions
"""

import logging
from typing import List, Optional, Tuple
from datetime import datetime
import pandas as pd
from selenium.webdriver.remote.webelement import WebElement

from .abstract_scraper_classes import (
    AbstractBoxscoreScraper, 
    AbstractPageScraper,
)
from ..data_access.abstract_data_access import AbstractDataAccess
from ..config.config import AbstractConfig
from ..error_handling.custom_exceptions import (
    ScrapingError, DataExtractionError, DataProcessingError, DataValidationError,
    ConfigurationError, ElementNotFoundError
)
from ..logging.logging_utils import log_performance, log_context, structured_log

logger = logging.getLogger(__name__)

class BoxscoreScraper(AbstractBoxscoreScraper):
    """
    A class for scraping NBA boxscore data.

    This class provides methods to scrape boxscores for multiple seasons and stat types.

    Attributes:
        config (AbstractConfig): Configuration object.
        data_access (AbstractDataAccess): Data access object.
        page_scraper (AbstractPageScraper): An instance of PageScraper.
    """

    @log_performance
    def __init__(self, config: AbstractConfig, data_access: AbstractDataAccess, page_scraper: AbstractPageScraper) -> None:
        """
        Initialize the BoxscoreScraper with configuration, data access, and page scraper.

        Args:
            config (AbstractConfig): Configuration object.
            data_access (AbstractDataAccess): Data access object.
            page_scraper (AbstractPageScraper): Page scraper object.

        Raises:
            ConfigurationError: If there's an issue with the provided configuration or dependencies.
        """
        try:
            self.config = config
            self.data_access = data_access
            self.page_scraper = page_scraper



            structured_log(logger, logging.INFO, "BoxscoreScraper initialized successfully", 
                           page_scraper_type=type(page_scraper).__name__)
        except Exception as e:
            structured_log(logger, logging.ERROR, "Error initializing BoxscoreScraper", 
                           error_message=str(e),
                           error_type=type(e).__name__)
            raise ConfigurationError(f"Error initializing BoxscoreScraper: {str(e)}")

    @log_performance
    def scrape_and_save_all_boxscores(self, seasons: List[str], first_start_date: str) -> None:
        """
        Scrape and save boxscores for all stat types and specified seasons.

        Args:
            seasons (List[str]): A list of seasons to scrape.
            first_start_date (str): The start date for the first season.

        Raises:
            DataValidationError: If seasons list is empty or first_start_date is invalid.
            ScrapingError: If there's an error during scraping process.
        """
        if not seasons:
            raise DataValidationError("Seasons list cannot be empty")
        if not self._is_valid_date(first_start_date):
            raise DataValidationError(f"Invalid first_start_date: {first_start_date}")

        with log_context(operation="scrape_all_boxscores", seasons=seasons, start_date=first_start_date):
            #cycle through each stat type, then each season, then each sub-season type
            boxscores_dataframes = []
            file_names = []
            for stat_type in self.config.stat_types:
                try:
                    structured_log(logger, logging.INFO, f"Scraping {stat_type} stats", stat_type=stat_type)
                    new_games = self.scrape_stat_type(seasons, first_start_date, stat_type)
                    file_name = f"games_{stat_type}.csv"
                    boxscores_dataframes.append(new_games)
                    file_names.append(file_name)
                    structured_log(logger, logging.INFO, f"Successfully scraped and saved {stat_type} stats", 
                                   stat_type=stat_type, seasons_count=len(seasons))
                except ScrapingError as e:
                    structured_log(logger, logging.ERROR, f"Error scraping {stat_type} stats", 
                                   stat_type=stat_type, error_message=str(e))
                    raise
            self.data_access.save_dataframes(boxscores_dataframes, file_names)
    @log_performance
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
            DataValidationError: If stat_type is not supported.
            DataProcessingError: If there's an error processing the scraped data.
        """
        if stat_type not in self.config.stat_types:
            raise DataValidationError(f"Unsupported stat type: {stat_type}")

        try:
            with log_context(operation="scrape_stat_type", stat_type=stat_type):
                new_games = pd.DataFrame()
                start_date = first_start_date

                for season in seasons:

                    season_year = int(season[:4])    
                    end_date = f"{self.config.off_season_start_month}/01/{season_year+1}"
                    if datetime.strptime(end_date, "%m/%d/%Y").date() > datetime.now().date():
                        end_date = datetime.now().strftime("%m/%d/%Y")

                    structured_log(logger, logging.INFO, f"Scraping {stat_type} stats for {season}", 
                                   stat_type=stat_type, season=season, start_date=start_date, end_date=end_date)
                        
                    df_season = self.scrape_sub_seasons(str(season), str(start_date), str(end_date), stat_type)
                    new_games = pd.concat([new_games, df_season], axis=0)
                    start_date = f"{self.config.regular_season_start_month}/01/{season_year+1}" #update start date for next season

                structured_log(logger, logging.INFO, f"Successfully scraped {stat_type} stats for all seasons", 
                               stat_type=stat_type, seasons_count=len(seasons))
                return new_games
        except Exception as e:
            structured_log(logger, logging.ERROR, f"Error processing scraped data for {stat_type}", 
                           stat_type=stat_type, error_message=str(e))
            raise DataProcessingError(f"Error processing scraped data for {stat_type}: {str(e)}")

    @log_performance
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

        Raises:
            ScrapingError: If there's an error during the scraping process.
        """
        with log_context(operation="scrape_sub_seasons", season=season, start_date=start_date, end_date=end_date, stat_type=stat_type):
            structured_log(logger, logging.INFO, f"Scraping {season} from {start_date} to {end_date} for {stat_type} stats")
            
            all_sub_seasons = pd.DataFrame()
            sub_season_types = self._determine_sub_season_types(season,start_date, end_date)

            for sub_season_type in sub_season_types:
                try:
                    df = self.scrape_to_dataframe(Season=season, DateFrom=start_date, DateTo=end_date, stat_type=stat_type, season_type=sub_season_type)
                    if not df.empty:
                        all_sub_seasons = pd.concat([all_sub_seasons, df], axis=0)
                    structured_log(logger, logging.INFO, f"Successfully scraped {sub_season_type} for {season}", 
                                   sub_season_type=sub_season_type, season=season)
                except ScrapingError as e:
                    structured_log(logger, logging.ERROR, f"Error scraping {sub_season_type} for {season}", 
                                   sub_season_type=sub_season_type, season=season, error_message=str(e))
                    raise

            return all_sub_seasons
    
    def _determine_sub_season_types(self, season: str, start_date: str, end_date: str) -> List[str]:
        """
        Determine sub-season types based on start and end dates.

        Args:
            start_date (str): The start date of the season (format: "MM/DD/YYYY").
            end_date (str): The end date of the season (format: "MM/DD/YYYY").

        Returns:
            List[str]: A list of sub-season types.

        Raises:
            DataValidationError: If the dates are invalid or cannot be parsed.
        """
        try:

            structured_log(logger, logging.INFO, "Determining sub-season types", 
                           start_date=start_date, end_date=end_date)
            sub_season_types = []
            season_year = int(season[:4])
            start_date = datetime.strptime(start_date, "%m/%d/%Y")
            end_date = datetime.strptime(end_date, "%m/%d/%Y")
            play_in_date = datetime(season_year + 1, self.config.play_in_month, 1)

            structured_log(logger, logging.INFO, "Play-in date", 
                           play_in_date=str(play_in_date))

            if start_date < play_in_date and end_date < play_in_date:
                sub_season_types.append(self.config.regular_season_text)
            elif start_date > play_in_date and end_date > play_in_date:
                sub_season_types.append(self.config.playoffs_season_text)
            else:
                sub_season_types.append(self.config.regular_season_text)
                sub_season_types.append(self.config.playoffs_season_text)
                sub_season_types.append(self.config.play_in_season_text)

            structured_log(logger, logging.INFO, "Determined sub-season types", 
                           sub_season_types=sub_season_types)
            return sub_season_types
        except ValueError as e:
            structured_log(logger, logging.ERROR, "Invalid date format", 
                           start_date=start_date, end_date=end_date, error_message=str(e))
            raise DataValidationError(f"Invalid date format: {str(e)}")

    @log_performance
    def scrape_to_dataframe(self, Season: str, DateFrom: str = "NONE", DateTo: str = "NONE", stat_type: str = 'traditional', season_type: str = "Regular+Season") -> pd.DataFrame:
        """
        Scrape data and convert it to a DataFrame.

        Args:
            Season (str): The season to scrape.
            DateFrom (str, optional): The start date (format: "MM/DD/YYYY" or "NONE"). Defaults to "NONE".
            DateTo (str, optional): The end date (format: "MM/DD/YYYY" or "NONE"). Defaults to "NONE".
            stat_type (str, optional): The type of stats to scrape. Defaults to 'traditional'.
            season_type (str, optional): The type of season. Defaults to "Regular+Season".

        Returns:
            pd.DataFrame: A DataFrame containing scraped data.

        Raises:
            DataExtractionError: If there's an error extracting data from the table.
        """
        with log_context(operation="scrape_to_dataframe", Season=Season, DateFrom=DateFrom, DateTo=DateTo, stat_type=stat_type, season_type=season_type):
            try:
                data_table = self.scrape_boxscores_table(Season, DateFrom, DateTo, stat_type, season_type)
                
                if data_table is None:
                    structured_log(logger, logging.WARNING, f"No data found", 
                                   Season=Season, season_type=season_type, stat_type=stat_type)
                    return pd.DataFrame()
                
                df = self._convert_table_to_df(data_table)
                structured_log(logger, logging.INFO, "Successfully scraped and converted data to DataFrame", 
                               rows=len(df), columns=len(df.columns))
                return df
            except ElementNotFoundError as e:
                structured_log(logger, logging.ERROR, "Error extracting data", 
                               error_message=str(e))
                raise DataExtractionError(f"Error extracting data: {str(e)}")

    @log_performance
    def scrape_boxscores_table(self, Season: str, DateFrom: str = "NONE", DateTo: str = "NONE", stat_type: str = 'traditional', season_type: str = "Regular+Season") -> Optional[WebElement]:
        """
        Scrape the boxscores table from the NBA stats website.

        Args:
            Season (str): The season to scrape.
            DateFrom (str, optional): The start date (format: "MM/DD/YYYY" or "NONE"). Defaults to "NONE".
            DateTo (str, optional): The end date (format: "MM/DD/YYYY" or "NONE"). Defaults to "NONE".
            stat_type (str, optional): The type of stats to scrape. Defaults to 'traditional'.
            season_type (str, optional): The type of season. Defaults to "Regular+Season".

        Returns:
            Optional[WebElement]: A WebElement containing the scraped table, or None if not found.

        Raises:
            ScrapingError: If there's an error during the scraping process.
        """
        with log_context(operation="scrape_boxscores_table", Season=Season, DateFrom=DateFrom, DateTo=DateTo, stat_type=stat_type, season_type=season_type):
            nba_url = self._construct_nba_url(stat_type, season_type, Season, DateFrom, DateTo)
            structured_log(logger, logging.INFO, f"Scraping URL", url=nba_url)

            try:
                table = self.page_scraper.scrape_page_table(nba_url, self.config.table_class_name, self.config.pagination_class_name, self.config.dropdown_class_name)
                structured_log(logger, logging.INFO, "Successfully scraped boxscores table")
                return table
            except ElementNotFoundError as e:
                structured_log(logger, logging.ERROR, "Error scraping table", 
                               error_message=str(e))
                raise ScrapingError(f"Error scraping table: {str(e)}")

    def _convert_table_to_df(self, data_table: WebElement) -> pd.DataFrame:
        """
        Convert a WebElement table to a DataFrame.

        Args:
            data_table (WebElement): A WebElement containing the table data.

        Returns:
            pd.DataFrame: A DataFrame representation of the table.

        Raises:
            DataExtractionError: If there's an error extracting data from the table.
        """
        try:
            table_html = data_table.get_attribute('outerHTML')
            dfs = pd.read_html(table_html, header=0)
            df = pd.concat(dfs)

            team_id, game_id = self._extract_team_and_game_ids_boxscores(data_table)
            df[self.config.team_id_column] = team_id
            df[self.config.game_id_column] = game_id

            structured_log(logger, logging.INFO, "Successfully converted table to DataFrame", 
                           rows=len(df), columns=len(df.columns))
            return df
        except Exception as e:
            structured_log(logger, logging.ERROR, "Error converting table to DataFrame", 
                           error_message=str(e))
            raise DataExtractionError(f"Error converting table to DataFrame: {str(e)}")

    def _construct_nba_url(self, stat_type: str, season_type: str, Season: str, DateFrom: str, DateTo: str) -> str:
        """
        Construct the URL for NBA stats website based on given parameters.

        Args:
            stat_type (str): The type of stats to scrape.
            season_type (str): The type of season.
            Season (str): The season to scrape.
            DateFrom (str): The start date (format: "MM/DD/YYYY" or "NONE").
            DateTo (str): The end date (format: "MM/DD/YYYY" or "NONE").

        Returns:
            str: The constructed URL string.

        Raises:
            ConfigurationError: If there's an error constructing the URL.
        """
        try:
            base_url = f"{self.config.nba_boxscores_url}-{stat_type}" 
            nba_url = f"{base_url}?SeasonType={season_type}"
            
            if not Season:
                nba_url = f"{nba_url}&DateFrom={DateFrom}&DateTo={DateTo}"
            else:
                if DateFrom == "NONE" and DateTo == "NONE":
                    nba_url = f"{nba_url}&Season={Season}"
                else:
                    #if season is current season, then we don't need to specify the season in the url
                    today = datetime.now()
                    current_season = today.year if today.month >= self.config.regular_season_start_month else today.year - 1

                    if Season[:4] == str(current_season):
                        nba_url = f"{nba_url}&DateFrom={DateFrom}&DateTo={DateTo}"
                    else:
                        nba_url = f"{nba_url}&Season={Season}&DateFrom={DateFrom}&DateTo={DateTo}"
            
            structured_log(logger, logging.INFO, "Constructed NBA URL", url=nba_url)

            nba_url = nba_url.rstrip('\\').strip()
            
            return nba_url
        except Exception as e:
            structured_log(logger, logging.ERROR, "Error constructing NBA URL", 
                           error_message=str(e))
            raise ConfigurationError(f"Error constructing NBA URL: {str(e)}")

    def _extract_team_and_game_ids_boxscores(self, data_table: WebElement) -> Tuple[pd.Series, pd.Series]:
        """
        Extract team and game IDs from the boxscores table.

        Args:
            data_table (WebElement): A WebElement containing the table data.

        Returns:
            Tuple[pd.Series, pd.Series]: A tuple of Series containing team IDs and game IDs.

        Raises:
            DataExtractionError: If there's an error extracting team and game IDs.
        """
        try:
            links = self.page_scraper.get_elements_by_class(self.config.teams_and_games_class_name, data_table)       
            links_list = [i.get_attribute("href") for i in links]
        
            team_id = pd.Series([i[-10:] for i in links_list if ('stats' in i)])
            game_id = pd.Series([i[-10:] for i in links_list if ('/game/' in i)])
        
            structured_log(logger, logging.INFO, "Successfully extracted team and game IDs", 
                           team_ids_count=len(team_id), game_ids_count=len(game_id))
            return team_id, game_id
        except Exception as e:
            structured_log(logger, logging.ERROR, "Error extracting team and game IDs", 
                           error_message=str(e))
            raise DataExtractionError(f"Error extracting team and game IDs: {str(e)}")

    @staticmethod
    def _is_valid_date(date_string: str) -> bool:
        """
        Check if a given date string is valid.

        Args:
            date_string (str): The date string to validate (format: "MM/DD/YYYY").

        Returns:
            bool: True if the date is valid, False otherwise.
        """
        try:
            datetime.strptime(date_string, "%m/%d/%Y")
            return True
        except ValueError:
            return False

