"""main.py

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
import logging

from ..logging.logging_setup import setup_logging
from .utils import (
    get_start_date_and_seasons,
    validate_data,
    concatenate_scraped_data,
)
from .di_container import DIContainer

LOG_FILE = "webscraping.log"

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

    # Initialize the Dependency Injection container
    container = DIContainer()

    # Get instances from the container
    config = container.config()
    data_access = container.data_access()
    nba_scraper = container.nba_scraper()
    
    try:

        setup_logging(config, LOG_FILE)
        logger = logging.getLogger(__name__)
        logger.info("Starting web scraping process")

        # Determine start date and seasons to scrape data for
        logger.info("Retrieving start date and seasons to scrape data for")
        #first_start_date, seasons = get_start_date_and_seasons(config, data_access)
        first_start_date = "5/11/2024"
        seasons = ["2023-24", ]
        
        # Scrape Boxscores
        logger.info(f"Scraping data from {first_start_date} for seasons: {seasons}")
        nba_scraper.scrape_and_save_all_boxscores(seasons, first_start_date)

        # Scrape Schedule for Matchups
        search_day = datetime.today().strftime('%A, %B %d')[:3]
        logger.info(f"Scraping matchups for {search_day}")
        nba_scraper.scrape_and_save_matchups_for_day(search_day)
 
        # Validate newly scraped data
        logger.info("Validating newly scraped data")
        validate_data(config, data_access, cumulative=False)

        # Combine newly scraped data with cumulative scraped data
        logger.info("Concatenating newly scraped data with cumulative scraped data")
        #concatenate_scraped_data(config, data_access)
        
        # Validate cumulative data
        logger.info("Validating cumulative scraped data")
        validate_data(config, data_access, cumulative=True)

        logger.info("Web scraping process completed successfully")

    except Exception as e:
        logger.error(f"An error occurred during the web scraping process: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()