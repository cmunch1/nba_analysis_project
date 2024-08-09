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
from datetime import date
from selenium.webdriver.remote.webdriver import WebDriver

from .boxscore_scraper import BoxscoreScraper
from .schedule_scraper import ScheduleScraper
from .web_driver_factory import WebDriverFactory
from ..config.config import config

class NbaScraper:
    """
    A facade class that combines boxscore and schedule scraping functionality for NBA data.

    This class delegates scraping tasks to specialized scraper classes and manages
    the WebDriver lifecycle using WebDriverFactory.

    Attributes:
        boxscore_scraper (Optional[BoxscoreScraper]): An instance of BoxscoreScraper.
        schedule_scraper (Optional[ScheduleScraper]): An instance of ScheduleScraper.
        driver (Optional[WebDriver]): The Selenium WebDriver instance.
        driver_factory (WebDriverFactory): An instance of WebDriverFactory.
    """

    def __init__(self):
        """
        Initialize the NbaScraper with a WebDriverFactory instance.
        """
        self.boxscore_scraper: Optional[BoxscoreScraper] = None
        self.schedule_scraper: Optional[ScheduleScraper] = None
        self.driver: Optional[WebDriver] = None
        self.driver_factory: WebDriverFactory = WebDriverFactory()

    def __enter__(self) -> 'NbaScraper':
        """
        Enter the runtime context related to this object.

        Creates the WebDriver using WebDriverFactory and initializes the scraper instances.

        Returns:
            NbaScraper: The NbaScraper instance.

        Raises:
            RuntimeError: If WebDriver creation fails.
        """
        try:
            self.driver = self.driver_factory.create_driver("chrome", options=config.webdriver_options)
            self.boxscore_scraper = BoxscoreScraper(self.driver)
            self.schedule_scraper = ScheduleScraper(self.driver)
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to initialize NbaScraper: {str(e)}")

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """
        Exit the runtime context related to this object.

        Closes the WebDriver and handles any exceptions that occurred in the context.

        Args:
            exc_type: The exception type if an exception was raised in the context.
            exc_value: The exception value if an exception was raised in the context.
            traceback: The traceback if an exception was raised in the context.

        Returns:
            bool: False to propagate exceptions, True to suppress them.
        """
        if self.driver:
            self.driver.quit()

        if exc_type is not None:
            print(f"An error occurred: {exc_type}, {exc_value}")
            # Log the error or perform any necessary error handling
            return False  # Propagate the exception
        return True

    def scrape_and_save_all_boxscores(self, seasons: List[str], first_start_date: date) -> None:
        """
        Scrape and save all boxscores for the given seasons.

        Args:
            seasons (List[str]): A list of seasons to scrape (e.g., ["2021-22", "2022-23"]).
            first_start_date (date): The start date for the first season.

        Raises:
            ValueError: If the boxscore_scraper is not initialized.
            RuntimeError: If scraping fails for any reason.
        """
        if not self.boxscore_scraper:
            raise ValueError("BoxscoreScraper is not initialized. Use NbaScraper as a context manager.")

        try:
            self.boxscore_scraper.scrape_and_save_all_boxscores(seasons, first_start_date)
        except Exception as e:
            raise RuntimeError(f"Failed to scrape and save boxscores: {str(e)}")

    def scrape_and_save_matchups_for_day(self, search_day: date) -> None:
        """
        Scrape and save matchups for a specific day.

        Args:
            search_day (date): The day to search for matchups.

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