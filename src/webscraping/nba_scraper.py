from .boxscore_scraper import BoxscoreScraper
from .schedule_scraper import ScheduleScraper
from selenium import webdriver

class NbaScraper:
    """
    A class that combines boxscore and schedule scraping functionality for NBA data.

    This class acts as a facade, delegating scraping tasks to specialized scraper classes.

    Attributes:
        driver: A Selenium WebDriver instance.
        boxscore_scraper: An instance of BoxscoreScraper.
        schedule_scraper: An instance of ScheduleScraper.
    """

    def __init__(self, driver):
        """
        Initialize the NbaScraper with a WebDriver and create instances of specialized scrapers.

        Args:
            driver: A Selenium WebDriver instance.
        """
        self.driver = driver
        self.boxscore_scraper = BoxscoreScraper(driver)
        self.schedule_scraper = ScheduleScraper(driver)

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
            False to propagate exceptions.
        """
        self.driver.quit()
        if exc_type is not None:
            print(f"An error occurred: {exc_type}, {exc_value}")
        return False

    def scrape_and_save_all_boxscores(self, seasons, first_start_date):
        """
        Scrape and save all boxscores for the given seasons.

        Args:
            seasons: A list of seasons to scrape.
            first_start_date: The start date for the first season.
        """
        self.boxscore_scraper.scrape_and_save_all_boxscores(seasons, first_start_date)

    def scrape_and_save_matchups_for_day(self, search_day):
        """
        Scrape and save matchups for a specific day.

        Args:
            search_day: The day to search for matchups.
        """
        self.schedule_scraper.scrape_and_save_matchups_for_day(search_day)


