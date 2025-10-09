"""
validation_scraper.py

This module provides functionality for scraping validation data from basketball-reference.com
to verify NBA matchup designations. It creates a validation dataset that can be used by the
data processing module to detect and correct mislabeled home/visitor team designations.

Key features:
- Scrapes game data from basketball-reference.com using date-based approach
- Parses scoreboard pages to extract all games for a given date
- Uses ordinal positioning to identify visitor (first) and home (second) teams
- Verifies home team using box score link URL
- Saves validation data for later comparison

Date-based approach benefits:
- One page load per date instead of per game (99% fewer requests)
- No reliance on potentially mislabeled NBA.com MATCH UP data
- Basketball-reference is authoritative source for home/visitor designation
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement

from .base_scraper_classes import BasePageScraper
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.framework.data_access.base_data_access import BaseDataAccess
from ml_framework.core.error_handling.error_handler_factory import ErrorHandlerFactory
from ml_framework.core.app_logging import log_performance, log_context, AppLogger

class ValidationScraper:
    """
    A class for scraping validation data from basketball-reference.com using date-based approach.

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
    def scrape_and_save_validation_data(self, dates: List[str]) -> None:
        """
        Scrape validation data for a list of dates and save to CSV file.

        Args:
            dates (List[str]): List of dates in MM/DD/YYYY format

        Raises:
            DataStorageError: If there's an error saving the data
        """
        try:
            self.app_logger.structured_log(logging.INFO,
                                         "Starting scrape and save validation data",
                                         date_count=len(dates))

            validation_df = self.scrape_validation_data_by_dates(dates)

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

    @log_performance
    def scrape_validation_data_by_dates(self, dates: List[str]) -> pd.DataFrame:
        """
        Scrape validation data for a list of dates from basketball-reference.com.

        Uses date-based scoreboard pages to get all games for each date.

        Args:
            dates (List[str]): List of dates in MM/DD/YYYY format

        Returns:
            pd.DataFrame: Validation data with columns:
                - DATE: Game date (MM/DD/YYYY)
                - VISITOR_TEAM_ABBREV: Visitor team abbreviation
                - VISITOR_TEAM_ID: Visitor team NBA ID
                - VISITOR_SCORE: Visitor team final score
                - HOME_TEAM_ABBREV: Home team abbreviation
                - HOME_TEAM_ID: Home team NBA ID
                - HOME_SCORE: Home team final score
                - BBREF_GAME_ID: Basketball-reference game ID (YYYYMMDD0HHH)
                - SOURCE: Data source (always 'basketball-reference')
                - SCRAPED_AT: Timestamp when scraped

        Raises:
            ScrapingError: If there's an error during the scraping process
        """
        with log_context(operation="scrape_validation_data_by_dates", date_count=len(dates)):
            try:
                self.app_logger.structured_log(logging.INFO, "Starting validation data scraping",
                                             date_count=len(dates))

                validation_records = []
                failed_dates = []

                # Get rate limiting config
                delay = getattr(self.config, 'rate_limiting', {}).get('delay_between_dates', 1.0)

                for date_str in dates:
                    try:
                        date_games = self._scrape_games_for_date(date_str)
                        if date_games:
                            validation_records.extend(date_games)
                            self.app_logger.structured_log(logging.DEBUG,
                                                         "Scraped games for date",
                                                         date=date_str,
                                                         game_count=len(date_games))
                        else:
                            self.app_logger.structured_log(logging.DEBUG,
                                                         "No games found for date",
                                                         date=date_str)

                        # Rate limiting - be respectful to basketball-reference.com
                        time.sleep(delay)

                    except Exception as e:
                        self.app_logger.structured_log(logging.WARNING,
                                                     "Failed to scrape validation data for date",
                                                     date=date_str,
                                                     error=str(e))
                        failed_dates.append(date_str)

                if failed_dates:
                    self.app_logger.structured_log(logging.WARNING,
                                                 "Some dates failed validation scraping",
                                                 failed_count=len(failed_dates),
                                                 failed_dates=failed_dates[:10])  # Log first 10

                df = pd.DataFrame(validation_records)

                self.app_logger.structured_log(logging.INFO,
                                             "Validation data scraping completed",
                                             successful_games=len(validation_records),
                                             failed_dates=len(failed_dates))

                return df

            except Exception as e:
                if hasattr(e, 'app_logger'):
                    raise
                self.app_logger.structured_log(logging.ERROR,
                                             "Error in scrape_validation_data_by_dates",
                                             error_message=str(e))
                raise self.error_handler.create_error_handler('scraping',
                    f"Error scraping validation data: {str(e)}")

    def _scrape_games_for_date(self, date_str: str) -> List[Dict]:
        """
        Scrape all games for a specific date from basketball-reference scoreboard.

        Args:
            date_str (str): Date in MM/DD/YYYY format

        Returns:
            List[Dict]: List of game validation records for this date
        """
        try:
            url = self._build_date_url(date_str)

            self.app_logger.structured_log(logging.DEBUG, "Loading scoreboard page",
                                         date=date_str, url=url)

            if not self.page_scraper.go_to_url(url):
                self.app_logger.structured_log(logging.WARNING, "Failed to load scoreboard page",
                                             date=date_str, url=url)
                return []

            # Parse all games from the scoreboard page
            games = self._parse_scoreboard_page(date_str)

            return games

        except Exception as e:
            self.app_logger.structured_log(logging.WARNING,
                                         "Error scraping games for date",
                                         date=date_str,
                                         error=str(e))
            return []

    def _build_date_url(self, date_str: str) -> str:
        """
        Build basketball-reference scoreboard URL for a specific date.

        Args:
            date_str (str): Date in MM/DD/YYYY format

        Returns:
            str: URL for scoreboard page (e.g., /boxscores/?month=10&day=31&year=2006)
        """
        date_obj = datetime.strptime(date_str, '%m/%d/%Y')
        base_url = self.config.scoreboard_base_url

        url = f"{base_url}?month={date_obj.month:02d}&day={date_obj.day:02d}&year={date_obj.year}"

        return url

    def _parse_scoreboard_page(self, date_str: str) -> List[Dict]:
        """
        Parse all games from basketball-reference scoreboard page.

        Basketball-reference structure:
        <div>
            <a href="/teams/PHO/2007.html">Phoenix</a>
            106                                          (text node)
            <span>Final</span>
            <a href="/teams/LAL/2007.html">LA Lakers</a>
            114                                          (text node)
            ...
            <a href="/boxscores/200610310LAL.html">Box Score</a>
        </div>

        Strategy:
        1. Find game containers (divs with box score links)
        2. For each game, extract team links, parse text content for scores

        Args:
            date_str (str): Date in MM/DD/YYYY format (for record keeping)

        Returns:
            List[Dict]: List of game validation records
        """
        try:
            games = []

            # Get all box score links - each represents one game
            box_score_links = self.page_scraper.page_scraper.driver.find_elements(
                By.CSS_SELECTOR, self.config.selectors.box_score_link
            )

            if not box_score_links:
                self.app_logger.structured_log(logging.DEBUG, "No games found on scoreboard",
                                             date=date_str)
                return []

            self.app_logger.structured_log(logging.DEBUG, "Found box score links",
                                         count=len(box_score_links))

            # Process each game
            for box_score_link in box_score_links:
                try:
                    # Get the parent game container
                    game_container = box_score_link.find_element(By.XPATH, "./..")

                    game_data = self._parse_single_game_from_container(
                        date_str,
                        game_container,
                        box_score_link
                    )

                    if game_data:
                        games.append(game_data)

                except Exception as e:
                    self.app_logger.structured_log(logging.DEBUG,
                                                 "Error parsing game from scoreboard",
                                                 date=date_str,
                                                 error=str(e))
                    continue

            return games

        except Exception as e:
            self.app_logger.structured_log(logging.WARNING,
                                         "Error parsing scoreboard page",
                                         date=date_str,
                                         error=str(e))
            return []

    def _parse_single_game_from_container(self, date_str: str, game_container: WebElement,
                                          box_score_link: WebElement) -> Optional[Dict]:
        """
        Parse a single game from its container div.

        Container structure:
            <div>
                <a href="/teams/PHO/2007.html">Phoenix</a>
                106
                <span>Final</span>
                <a href="/teams/LAL/2007.html">LA Lakers</a>
                114
                ...
                <a href="/boxscores/200610310LAL.html">Box Score</a>
            </div>

        Strategy: Parse text content, extract scores using regex between team names

        Args:
            date_str (str): Date in MM/DD/YYYY format
            game_container (WebElement): Container div for the game
            box_score_link (WebElement): Box score link element

        Returns:
            Optional[Dict]: Game validation record or None if parsing failed
        """
        try:
            import re

            # Get team links within this container
            team_links = game_container.find_elements(By.CSS_SELECTOR, self.config.selectors.team_links)

            if len(team_links) < 2:
                self.app_logger.structured_log(logging.DEBUG,
                                             "Not enough team links in container",
                                             team_links_count=len(team_links))
                return None

            # Extract visitor team (first team link - appears twice, we want the first occurrence)
            visitor_href = team_links[0].get_attribute('href')
            visitor_abbrev = self._extract_team_abbrev_from_url(visitor_href)

            # Extract home team (second unique team link)
            # Find the first team link that's different from visitor
            home_href = None
            for link in team_links[1:]:
                href = link.get_attribute('href')
                if href != visitor_href:
                    home_href = href
                    break

            if not home_href:
                self.app_logger.structured_log(logging.DEBUG,
                                             "Could not find distinct home team link")
                return None

            home_abbrev = self._extract_team_abbrev_from_url(home_href)

            # Extract scores from container text using regex
            # Pattern: team name, then digits, then "Final", then team name, then digits
            container_text = game_container.text

            # Find all sequences of digits
            scores = re.findall(r'\b(\d{2,3})\b', container_text)

            if len(scores) < 2:
                self.app_logger.structured_log(logging.DEBUG,
                                             "Could not extract scores from container",
                                             container_text=container_text[:100])
                return None

            # First two numeric values should be the final scores
            visitor_score = scores[0]
            home_score = scores[1]

            # Extract basketball-reference game ID from box score link
            box_score_href = box_score_link.get_attribute('href')
            bbref_game_id = box_score_href.split('/')[-1].replace('.html', '')

            # Verify home team matches box score link (should end with home team abbrev)
            if self.config.parsing.verify_home_team_from_url:
                if not bbref_game_id.endswith(home_abbrev):
                    self.app_logger.structured_log(logging.WARNING,
                                                 "Home team verification failed",
                                                 date=date_str,
                                                 home_abbrev=home_abbrev,
                                                 bbref_game_id=bbref_game_id)
                    if self.config.error_handling.fail_on_verification_mismatch:
                        return None

            # Convert team abbreviations to NBA team IDs
            visitor_team_id = self.team_abbrev_to_id.get(visitor_abbrev.upper())
            home_team_id = self.team_abbrev_to_id.get(home_abbrev.upper())

            if not visitor_team_id or not home_team_id:
                self.app_logger.structured_log(logging.DEBUG,
                                             "Unknown team abbreviation",
                                             date=date_str,
                                             visitor_abbrev=visitor_abbrev,
                                             home_abbrev=home_abbrev)
                if self.config.error_handling.warn_on_unknown_teams:
                    self.app_logger.structured_log(logging.WARNING,
                                                 "Unknown team in validation data",
                                                 visitor=visitor_abbrev,
                                                 home=home_abbrev)
                return None

            game_data = {
                'DATE': date_str,
                'VISITOR_TEAM_ABBREV': visitor_abbrev,
                'VISITOR_TEAM_ID': visitor_team_id,
                'VISITOR_SCORE': visitor_score,
                'HOME_TEAM_ABBREV': home_abbrev,
                'HOME_TEAM_ID': home_team_id,
                'HOME_SCORE': home_score,
                'BBREF_GAME_ID': bbref_game_id,
                'SOURCE': 'basketball-reference',
                'SCRAPED_AT': datetime.now().isoformat()
            }

            return game_data

        except Exception as e:
            self.app_logger.structured_log(logging.DEBUG,
                                         "Error parsing single game from container",
                                         date=date_str,
                                         error=str(e))
            return None

    def _extract_team_abbrev_from_url(self, url: str) -> str:
        """
        Extract team abbreviation from basketball-reference team URL.

        Args:
            url (str): Team URL (e.g., /teams/LAL/2007.html)

        Returns:
            str: Team abbreviation (e.g., 'LAL')
        """
        # URL format: /teams/LAL/2007.html
        # Split by '/' and get the element at configured index
        parts = url.split('/')
        abbrev_index = self.config.team_abbrev.url_path_index

        if len(parts) > abbrev_index:
            return parts[abbrev_index]

        return ''
