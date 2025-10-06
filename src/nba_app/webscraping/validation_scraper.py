"""
validation_scraper.py

This module provides functionality to scrape validation data from basketball-reference.com
to cross-reference and validate NBA.com data, particularly for home/visitor team designations.

Key features:
- Scrapes minimal validation dataset (game_id, home_team, visitor_team, scores)
- Uses basketball-reference.com as authoritative source
- Lightweight and fast - only scrapes essential validation fields
- Integrates with existing PageScraper infrastructure
"""

import logging
from typing import List, Dict, Optional, Tuple
import pandas as pd
from datetime import datetime
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement

from .base_scraper_classes import BasePageScraper
from ml_framework.framework.data_access.base_data_access import BaseDataAccess
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.error_handling.error_handler_factory import ErrorHandlerFactory
from ml_framework.core.app_logging import log_performance, log_context, AppLogger

class ValidationScraper:
    """
    A class for scraping validation data from basketball-reference.com.

    This scraper collects minimal data needed to validate NBA.com scraped data,
    focusing on home/visitor team designations and basic game information.

    Attributes:
        config (BaseConfigManager): Configuration object
        data_access (BaseDataAccess): Data access object
        page_scraper (BasePageScraper): Page scraper object
        app_logger (AppLogger): Application logger instance
        error_handler (ErrorHandlerFactory): Error handler factory instance
        team_id_to_abbrev (Dict[str, str]): NBA team ID to basketball-reference abbreviation mapping
        team_abbrev_to_id (Dict[str, str]): Basketball-reference abbreviation to NBA team ID mapping
    """

    @log_performance
    def __init__(self, config: BaseConfigManager, data_access: BaseDataAccess,
                 page_scraper: BasePageScraper, app_logger: AppLogger,
                 error_handler: ErrorHandlerFactory) -> None:
        """
        Initialize the ValidationScraper.

        Args:
            config (BaseConfigManager): Configuration object
            data_access (BaseDataAccess): Data access object
            page_scraper (BasePageScraper): Page scraper object
            app_logger (AppLogger): Application logger instance
            error_handler (ErrorHandlerFactory): Error handler factory instance
        """
        self.config = config
        self.data_access = data_access
        self.page_scraper = page_scraper
        self.app_logger = app_logger
        self.error_handler = error_handler

        # Load team mappings from config (automatically loaded by config manager)
        self.team_id_to_abbrev, self.team_abbrev_to_id = self._load_team_mappings()

        self.app_logger.structured_log(logging.INFO, "ValidationScraper initialized",
                                      teams_mapped=len(self.team_id_to_abbrev))

    def _load_team_mappings(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Load team ID mappings from config attributes.

        The config system automatically loads team_mapping.yaml and exposes:
        - config.team_id_to_abbrev: Current NBA team mappings
        - config.historical_teams: Historical team abbreviations

        Returns:
            Tuple[Dict[str, str], Dict[str, str]]: (team_id_to_abbrev, team_abbrev_to_id)

        Raises:
            ConfigurationError: If team mappings are not found in config
        """
        try:
            # Get team mappings from config (loaded automatically from team_mapping.yaml)
            if not hasattr(self.config, 'team_id_to_abbrev'):
                raise self.error_handler.create_error_handler('configuration',
                    "team_id_to_abbrev not found in config. Check that team_mapping.yaml is in config directory.")

            # Config system converts dicts to SimpleNamespace, so we need to convert back
            team_id_to_abbrev_obj = self.config.team_id_to_abbrev
            historical_teams_obj = getattr(self.config, 'historical_teams', {})

            # Convert SimpleNamespace to dict
            team_id_to_abbrev = vars(team_id_to_abbrev_obj) if hasattr(team_id_to_abbrev_obj, '__dict__') else team_id_to_abbrev_obj
            historical_teams = vars(historical_teams_obj) if hasattr(historical_teams_obj, '__dict__') else historical_teams_obj

            # Build reverse mapping (abbrev -> id)
            team_abbrev_to_id = {abbrev: team_id for team_id, abbrev in team_id_to_abbrev.items()}

            # Add historical teams to abbrev -> id mapping
            team_abbrev_to_id.update(historical_teams)

            self.app_logger.structured_log(logging.INFO, "Team mappings loaded from config",
                                         current_teams=len(team_id_to_abbrev),
                                         historical_teams=len(historical_teams),
                                         total_abbrev_mappings=len(team_abbrev_to_id))

            if not team_id_to_abbrev:
                raise self.error_handler.create_error_handler('configuration',
                    "No team mappings found in config")

            return team_id_to_abbrev, team_abbrev_to_id

        except Exception as e:
            if hasattr(e, 'app_logger'):
                raise
            self.app_logger.structured_log(logging.ERROR, "Error loading team mappings from config",
                                         error_message=str(e))
            raise self.error_handler.create_error_handler('configuration',
                f"Failed to load team mappings: {str(e)}")

    @log_performance
    def scrape_validation_data_for_games(self, game_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Scrape validation data for a list of games from basketball-reference.com.

        Args:
            game_metadata (pd.DataFrame): DataFrame with columns:
                - GAME_ID: NBA game ID
                - GAME_DATE: Game date (MM/DD/YYYY format)
                - HOME_TEAM_ID: Home team NBA ID (team with "vs." in matchup)

        Returns:
            pd.DataFrame: Validation data with columns:
                - GAME_ID: NBA game ID
                - HOME_TEAM_ID: Home team NBA ID
                - VISITOR_TEAM_ID: Visitor team NBA ID
                - HOME_SCORE: Home team final score
                - VISITOR_SCORE: Visitor team final score
                - SOURCE: Data source (always 'basketball-reference')
                - SCRAPED_AT: Timestamp when scraped

        Raises:
            ScrapingError: If there's an error during the scraping process
        """
        with log_context(operation="scrape_validation_data", game_count=len(game_metadata)):
            try:
                self.app_logger.structured_log(logging.INFO, "Starting validation data scraping",
                                             game_count=len(game_metadata))

                validation_records = []
                failed_games = []

                for _, row in game_metadata.iterrows():
                    game_id = row['GAME_ID']
                    game_date = row['GAME_DATE']
                    home_team_id = row['HOME_TEAM_ID']

                    try:
                        validation_data = self._scrape_single_game_validation(
                            game_id, game_date, home_team_id
                        )
                        if validation_data:
                            validation_records.append(validation_data)
                        else:
                            failed_games.append(game_id)
                    except Exception as e:
                        self.app_logger.structured_log(logging.WARNING,
                                                     "Failed to scrape validation data for game",
                                                     game_id=game_id,
                                                     error=str(e))
                        failed_games.append(game_id)

                if failed_games:
                    self.app_logger.structured_log(logging.WARNING,
                                                 "Some games failed validation scraping",
                                                 failed_count=len(failed_games),
                                                 failed_games=failed_games[:10])  # Log first 10

                df = pd.DataFrame(validation_records)

                self.app_logger.structured_log(logging.INFO,
                                             "Validation data scraping completed",
                                             successful_count=len(validation_records),
                                             failed_count=len(failed_games))

                return df

            except Exception as e:
                if hasattr(e, 'app_logger'):
                    raise
                self.app_logger.structured_log(logging.ERROR,
                                             "Error in scrape_validation_data_for_games",
                                             error_message=str(e))
                raise self.error_handler.create_error_handler('scraping',
                    f"Error scraping validation data: {str(e)}")

    @log_performance
    def _scrape_single_game_validation(self, game_id: str, game_date: str,
                                       home_team_id: str) -> Optional[Dict[str, str]]:
        """
        Scrape validation data for a single game from basketball-reference.com.

        Basketball-reference uses game IDs in format: YYYYMMDD0HHH
        where YYYYMMDD is the date and HHH is the home team abbreviation

        Args:
            game_id (str): NBA game ID (e.g., '20600001')
            game_date (str): Game date in MM/DD/YYYY format (e.g., '10/31/2006')
            home_team_id (str): NBA team ID of home team (e.g., '1610612748')

        Returns:
            Optional[Dict]: Validation data dictionary or None if scraping failed
        """
        try:
            # Convert date from MM/DD/YYYY to YYYYMMDD
            from datetime import datetime
            date_obj = datetime.strptime(game_date, '%m/%d/%Y')
            date_str = date_obj.strftime('%Y%m%d')

            # Convert team ID to basketball-reference abbreviation
            home_team_abbrev = self.team_id_to_abbrev.get(home_team_id)
            if not home_team_abbrev:
                self.app_logger.structured_log(logging.WARNING,
                                             "Unknown home team ID for game",
                                             game_id=game_id,
                                             home_team_id=home_team_id)
                return None

            # Construct basketball-reference game ID and URL
            bbref_game_id = f"{date_str}0{home_team_abbrev}"
            url = f"https://www.basketball-reference.com/boxscores/{bbref_game_id}.html"

            self.app_logger.structured_log(logging.DEBUG, "Attempting to scrape validation data",
                                         game_id=game_id,
                                         bbref_game_id=bbref_game_id,
                                         url=url)

            if not self.page_scraper.go_to_url(url):
                self.app_logger.structured_log(logging.WARNING, "Failed to load page",
                                             game_id=game_id, url=url)
                return None

            # Extract visitor and home team info
            visitor_team_abbrev, visitor_score = self._extract_visitor_info()
            home_team_abbrev, home_score = self._extract_home_info()

            if not all([visitor_team_abbrev, home_team_abbrev]):
                self.app_logger.structured_log(logging.WARNING,
                                             "Failed to extract team info",
                                             game_id=game_id)
                return None

            # Convert team abbreviations to NBA team IDs
            visitor_team_id = self.team_abbrev_to_id.get(visitor_team_abbrev.upper())
            home_team_id = self.team_abbrev_to_id.get(home_team_abbrev.upper())

            if not visitor_team_id or not home_team_id:
                self.app_logger.structured_log(logging.WARNING,
                                             "Unknown team abbreviation",
                                             game_id=game_id,
                                             visitor=visitor_team_abbrev,
                                             home=home_team_abbrev)
                return None

            validation_data = {
                'GAME_ID': game_id,
                'HOME_TEAM_ID': home_team_id,
                'VISITOR_TEAM_ID': visitor_team_id,
                'HOME_SCORE': home_score,
                'VISITOR_SCORE': visitor_score,
                'SOURCE': 'basketball-reference',
                'SCRAPED_AT': datetime.now().isoformat()
            }

            self.app_logger.structured_log(logging.DEBUG, "Successfully scraped validation data",
                                         game_id=game_id,
                                         home_team=home_team_abbrev,
                                         visitor_team=visitor_team_abbrev)

            return validation_data

        except Exception as e:
            self.app_logger.structured_log(logging.WARNING,
                                         "Error scraping single game validation",
                                         game_id=game_id,
                                         error=str(e))
            return None

    def _extract_visitor_info(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract visitor team abbreviation and score from basketball-reference page.

        Returns:
            Tuple[Optional[str], Optional[str]]: (team_abbreviation, score)
        """
        try:
            # Basketball-reference has a specific structure
            # The visitor (away) team is typically in the first scorebox
            scorebox = self.page_scraper.page_scraper.driver.find_element(
                By.CSS_SELECTOR, "div.scorebox"
            )

            # First team is visitor
            visitor_elem = scorebox.find_element(By.CSS_SELECTOR, "div:first-child strong a")
            visitor_abbrev = visitor_elem.get_attribute("href").split("/")[-2].upper()

            # Score is in the score div
            visitor_score_elem = scorebox.find_element(
                By.CSS_SELECTOR, "div:first-child div.scores div.score"
            )
            visitor_score = visitor_score_elem.text.strip()

            return visitor_abbrev, visitor_score

        except Exception as e:
            self.app_logger.structured_log(logging.DEBUG, "Error extracting visitor info",
                                         error=str(e))
            return None, None

    def _extract_home_info(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract home team abbreviation and score from basketball-reference page.

        Returns:
            Tuple[Optional[str], Optional[str]]: (team_abbreviation, score)
        """
        try:
            # Basketball-reference has a specific structure
            # The home team is typically in the second scorebox
            scorebox = self.page_scraper.page_scraper.driver.find_element(
                By.CSS_SELECTOR, "div.scorebox"
            )

            # Second team is home
            home_elem = scorebox.find_element(By.CSS_SELECTOR, "div:nth-child(2) strong a")
            home_abbrev = home_elem.get_attribute("href").split("/")[-2].upper()

            # Score is in the score div
            home_score_elem = scorebox.find_element(
                By.CSS_SELECTOR, "div:nth-child(2) div.scores div.score"
            )
            home_score = home_score_elem.text.strip()

            return home_abbrev, home_score

        except Exception as e:
            self.app_logger.structured_log(logging.DEBUG, "Error extracting home info",
                                         error=str(e))
            return None, None

    @log_performance
    def scrape_and_save_validation_data(self, game_metadata: pd.DataFrame) -> None:
        """
        Scrape validation data and save to CSV file.

        Args:
            game_metadata (pd.DataFrame): DataFrame with columns:
                - GAME_ID: NBA game ID
                - GAME_DATE: Game date (MM/DD/YYYY format)
                - HOME_TEAM_ID: Home team NBA ID

        Raises:
            DataStorageError: If there's an error saving the data
        """
        try:
            self.app_logger.structured_log(logging.INFO,
                                         "Starting scrape and save validation data",
                                         game_count=len(game_metadata))

            validation_df = self.scrape_validation_data_for_games(game_metadata)

            if validation_df.empty:
                self.app_logger.structured_log(logging.WARNING,
                                             "No validation data scraped - skipping save")
                return

            # Save to configured location
            file_name = self.config.validation_data_file
            self.data_access.save_dataframes([validation_df], [file_name])

            self.app_logger.structured_log(logging.INFO,
                                         "Validation data saved successfully",
                                         file_name=file_name,
                                         record_count=len(validation_df))

        except Exception as e:
            if hasattr(e, 'app_logger'):
                raise
            self.app_logger.structured_log(logging.ERROR,
                                         "Error in scrape_and_save_validation_data",
                                         error_message=str(e))
            raise self.error_handler.create_error_handler('data_storage',
                f"Error saving validation data: {str(e)}")
