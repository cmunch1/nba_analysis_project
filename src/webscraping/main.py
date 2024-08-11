"""main_webscraper.py

This module serves as the main entry point for the NBA web scraping application.
It orchestrates the process of scraping box scores and matchups from NBA websites,
validating the scraped data, and concatenating it with existing data.

Key features:
- Scrapes box scores for multiple seasons
- Scrapes matchups for the current day
- Validates scraped data
- Concatenates new data with existing data
- Implements logging for better debugging and monitoring
"""

from datetime import datetime
from typing import Tuple, List
import logging

from ..logging.logging_setup import setup_logging

from .utils import (
    get_start_date_and_seasons,
    validate_data,
    concatenate_scraped_data,
)


# Dependency Injection Setup
from .di_container import DIContainer
container = DIContainer()


# Set up logging
setup_logging("webscraping.log")
logger = logging.getLogger(__name__)

def main() -> None:
    """
    Runs the main functionality of the web scraping application.
    
    This function performs the following tasks:
    1. Retrieves the start date and seasons to scrape data for
    2. Initializes an NbaScraper instance and uses it to:
        a. Scrape and save all box scores for the specified seasons
        b. Scrape and save the matchups for today's date
    3. Validates the scraped data
    4. Concatenates newly scraped data with cumulative scraped data
    
    Raises:
        Exception: If any error occurs during the scraping process
    """
    try:
        logger.info("Starting web scraping process")
        
        #first_start_date, seasons = get_start_date_and_seasons()
        first_start_date, seasons = "5/11/2024", ["2023-24",] #test case
        
        logger.info(f"Scraping data from {first_start_date} for seasons: {seasons}")

        scraper = container.nba_scraper()
            
        # Scrape all boxscores for all seasons
        scraper.scrape_and_save_all_boxscores(seasons, first_start_date)
        logger.info("Finished scraping box scores")

        # Scrape schedule for today's matchups
        search_day = datetime.today().strftime('%A, %B %d')[:3]
        scraper.scrape_and_save_matchups_for_day(search_day)
        logger.info(f"Finished scraping matchups for {search_day}")

        # Validate newly scraped data
        validate_data(cumulative=False)
        logger.info("Validated newly scraped data")

        # Combine newly scraped data with cumulative scraped data
        #concatenate_scraped_data()
        logger.info("Concatenated new data with existing data")

        # Validate cumulative data
        validate_data(cumulative=True)
        logger.info("Validated cumulative data")

        logger.info("Web scraping process completed successfully")

    except Exception as e:
        logger.error(f"An error occurred during the web scraping process: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
