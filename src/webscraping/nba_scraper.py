from .boxscore_scraper import BoxscoreScraper
from .schedule_scraper import ScheduleScraper

from .utils import activate_web_driver



class NbaScraper:
    """
    A class that combines boxscore and schedule scraping functionality for NBA data.

    This class acts as a facade, delegating scraping tasks to specialized scraper classes.

    Attributes:
        boxscore_scraper: An instance of BoxscoreScraper.
        schedule_scraper: An instance of ScheduleScraper.
    """

    def __init__(self):
        """
        Initialize the NbaScraper with WebDriver instances for boxscore and schedule scraping.
        """
        self.boxscore_scraper = None
        self.schedule_scraper = None


    def __enter__(self):
        """
        Enter the runtime context related to this object.

        Returns:
            The NbaScraper instance.
        """
        self.driver = activate_web_driver("Chrome")
        self.boxscore_scraper = BoxscoreScraper(self.driver)
        self.schedule_scraper = ScheduleScraper(self.driver)
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


