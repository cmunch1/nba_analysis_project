"""
matchup_validator.py

This module provides functionality to validate and correct NBA matchup designations
by cross-referencing with alternative sources when the primary source has errors.

This became necessary in late 2024 when NBA.com started having issues with the matchup designations -
both teams are listed as home or visitor.

Key features:
- Validates home/visitor team designations in scraped data
- Fetches correct matchup data from alternative sources when needed
- Updates existing data files with corrections
"""

import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement

from .abstract_scraper_classes import AbstractPageScraper
from platform_core.core.config_management.base_config_manager import BaseConfigManager
from platform_core.framework.data_access.base_data_access import BaseDataAccess
from platform_core.core.error_handling.error_handler import (
    DataValidationError, DataExtractionError, DataStorageError
)
from platform_core.core.app_logging import log_performance, structured_log

logger = logging.getLogger(__name__)

class MatchupValidator:
    """
    A class for validating and correcting NBA matchup designations.
    
    Attributes:
        config (BaseConfigManager): Configuration object
        data_access (BaseDataAccess): Data access object
        page_scraper (AbstractPageScraper): Page scraper object
    """
    
    def __init__(self, config: BaseConfigManager, data_access: BaseDataAccess,
                 page_scraper: AbstractPageScraper) -> None:
        """
        Initialize the MatchupValidator.
        
        Args:
            config (BaseConfigManager): Configuration object
            data_access (BaseDataAccess): Data access object
            page_scraper (AbstractPageScraper): Page scraper object
        """
        self.config = config
        self.data_access = data_access
        self.page_scraper = page_scraper
        structured_log(logger, logging.INFO, "MatchupValidator initialized")

    @log_performance
    def validate_matchup_designations(self, table_element: WebElement) -> Tuple[bool, List[str]]:
        """
        Validate that each matchup has one home and one visitor team.
        
        Args:
            table_element (WebElement): The scraped table containing matchup data
            
        Returns:
            Tuple[bool, List[str]]: A tuple containing:
                - bool: True if all matchups are valid, False otherwise
                - List[str]: List of game IDs with invalid matchups
        """
        try:
            matchup_elements = table_element.find_elements(By.CSS_SELECTOR, "[class*='Crom_matchup']")
            invalid_game_ids = []
            
            for matchup in matchup_elements:
                matchup_text = matchup.text
                # Count occurrences of home (vs) and visitor (@) designations
                home_count = matchup_text.count('vs')
                visitor_count = matchup_text.count('@')
                
                if home_count != 1 or visitor_count != 1:
                    # Get the game ID for this matchup
                    game_id = matchup.find_element(By.CSS_SELECTOR, "[class*='Anchor_anchor']").get_attribute('href').split('/')[-1]
                    invalid_game_ids.append(game_id)
                    structured_log(logger, logging.WARNING, "Invalid matchup designation found",
                                 matchup_text=matchup_text,
                                 game_id=game_id,
                                 home_count=home_count,
                                 visitor_count=visitor_count)
            
            is_valid = len(invalid_game_ids) == 0
            return is_valid, invalid_game_ids
            
        except Exception as e:
            structured_log(logger, logging.ERROR, "Error validating matchup designations",
                          error_message=str(e))
            raise DataValidationError(f"Error validating matchup designations: {str(e)}")

    @log_performance
    def fetch_alternative_matchup_data(self, game_ids: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Fetch correct matchup data from basketball-reference.com for given game IDs.
        
        Args:
            game_ids (List[str]): List of game IDs needing correction
            
        Returns:
            Dict[str, Dict[str, str]]: Dictionary mapping game IDs to corrected matchup info
        """
        corrected_matchups = {}
        
        try:
            for game_id in game_ids:
                alternative_url = f"https://www.basketball-reference.com/boxscores/{game_id}.html"
                
                if self.page_scraper.go_to_url(alternative_url):
                    # Extract home and visitor teams
                    home_team = self._extract_home_team()
                    visitor_team = self._extract_visitor_team()
                    
                    corrected_matchups[game_id] = {
                        'home_team': home_team,
                        'visitor_team': visitor_team
                    }
                    
                    structured_log(logger, logging.INFO, "Retrieved corrected matchup data",
                                 game_id=game_id,
                                 home_team=home_team,
                                 visitor_team=visitor_team)
                    
            return corrected_matchups
            
        except Exception as e:
            structured_log(logger, logging.ERROR, "Error fetching alternative matchup data",
                          error_message=str(e))
            raise DataExtractionError(f"Error fetching alternative matchup data: {str(e)}")

    def _extract_home_team(self) -> str:
        """Extract home team from basketball-reference.com page."""
        try:
            # Implement specific extraction logic for basketball-reference.com
            # This is a placeholder - you'll need to implement the actual logic
            home_element = self.page_scraper.get_elements_by_class("home-team-class")[0]
            return home_element.text
        except Exception as e:
            raise DataExtractionError(f"Error extracting home team: {str(e)}")

    def _extract_visitor_team(self) -> str:
        """Extract visitor team from basketball-reference.com page."""
        try:
            # Implement specific extraction logic for basketball-reference.com
            # This is a placeholder - you'll need to implement the actual logic
            visitor_element = self.page_scraper.get_elements_by_class("visitor-team-class")[0]
            return visitor_element.text
        except Exception as e:
            raise DataExtractionError(f"Error extracting visitor team: {str(e)}")

    @log_performance
    def update_files_with_corrections(self, corrected_matchups: Dict[str, Dict[str, str]]) -> None:
        """
        Update the CSV files with corrected matchup information.
        
        Args:
            corrected_matchups (Dict[str, Dict[str, str]]): Dictionary of corrected matchup data
        """
        try:
            for stat_type in self.config.stat_types:
                file_path = f"games_{stat_type}.csv"
                df = self.data_access.load_dataframe(file_path)
                
                for game_id, matchup_info in corrected_matchups.items():
                    # Update the matchup column with corrected information
                    mask = df['GAME_ID'] == game_id
                    df.loc[mask, 'MATCHUP'] = df.loc[mask, 'TEAM_ID'].apply(
                        lambda x: f"{matchup_info['visitor_team']} @ {matchup_info['home_team']}"
                        if x == matchup_info['visitor_team']
                        else f"{matchup_info['home_team']} vs {matchup_info['visitor_team']}"
                    )
                
                # Save the updated dataframe
                self.data_access.save_dataframe(df, file_path)
                structured_log(logger, logging.INFO, "Updated file with corrections",
                             file_name=file_path,
                             corrections_count=len(corrected_matchups))
                
        except Exception as e:
            structured_log(logger, logging.ERROR, "Error updating files with corrections",
                          error_message=str(e))
            raise DataStorageError(f"Error updating files with corrections: {str(e)}") 