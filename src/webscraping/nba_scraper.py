"""
nba_scraper.py

This module provides a facade for scraping NBA data, including boxscores and schedules.

It combines the functionality of BoxscoreScraper and ScheduleScraper into a single interface,
making it easier to manage web scraping tasks for NBA data. The module uses Selenium WebDriver
for web interactions and implements context management for proper resource handling.

Key Classes:
    - NbaScraper: Main class that orchestrates NBA data scraping operations.

Dependencies:
    - Selenium WebDriver
    - BoxscoreScraper and ScheduleScraper from local modules
"""

from typing import List, Optional

from .abstract_scraper_classes import (
    AbstractNbaScraper,
    AbstractBoxscoreScraper, 
    AbstractScheduleScraper,
)

from ..config.config import AbstractConfig

class NbaScraper(AbstractNbaScraper):
    """
    A facade class that combines boxscore and schedule scraping functionality for NBA data.

    This class delegates scraping tasks to specialized scraper classes and manages
    the WebDriver lifecycle using WebDriverFactory.

    Attributes:
        boxscore_scraper (Optional[AbstractBoxscoreScraper]): An instance of BoxscoreScraper.
        schedule_scraper (Optional[AbstractScheduleScraper]): An instance of ScheduleScraper.

    """

    def __init__(self, config: AbstractConfig, boxscore_scraper: AbstractBoxscoreScraper, schedule_scraper: AbstractScheduleScraper):
        """
        Initialize the NbaScraper with a WebDriverFactory instance.
        """
        self.config = config
        self.boxscore_scraper = boxscore_scraper
        self.schedule_scraper =  schedule_scraper


    def scrape_and_save_all_boxscores(self, seasons: List[str], first_start_date: str) -> None:
        """
        Scrape and save all boxscores for the given seasons.

        Args:
            seasons (List[str]): A list of seasons to scrape (e.g., ["2021-22", "2022-23"]).
            first_start_date (str): The start date for the first season.

        Raises:
            ValueError: If the boxscore_scraper is not initialized.
            RuntimeError: If scraping fails for any reason.
        """
        if not self.boxscore_scraper:
            raise ValueError("BoxscoreScraper is not initialized.")

        try:
            self.boxscore_scraper.scrape_and_save_all_boxscores(seasons, first_start_date)
        except Exception as e:
            raise RuntimeError(f"Failed to scrape and save boxscores: {str(e)}")

    def scrape_and_save_matchups_for_day(self, search_day: str) -> None:
        """
        Scrape and save matchups for a specific day.

        Args:
            search_day (str): The day to search for matchups.

        Raises:
            ValueError: If the schedule_scraper is not initialized.
            RuntimeError: If scraping fails for any reason.
        """
        if not self.schedule_scraper:
            raise ValueError("ScheduleScraper is not initialized. Use NbaScraper as a context manager.")

        try:
            self.schedule_scraper.scrape_and_save_matchups_for_day(search_day)
        except Exception as e:
            raise RuntimeError(f"Failed to scrape and save matchups for {search_day}: {str(e)}")