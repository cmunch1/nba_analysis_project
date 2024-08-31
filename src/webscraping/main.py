"""main.py

This module serves as the main entry point for the NBA web scraping application.
It orchestrates the process of scraping box scores and matchups from NBA websites,
validating the scraped data, and concatenating it with existing data.

Key features:
- Scrapes box scores for multiple seasons
- Scrapes matchups for the current day
- Validates scraped data
- Concatenates new data with existing data
- Implements comprehensive logging for better debugging and monitoring
- Utilizes structured logging and context managers for consistent log formatting
- Implements granular error handling with custom exceptions
- Uses dependency injection for better modularity and testability
"""

import sys
import traceback
from datetime import datetime
import logging

from ..logging.logging_setup import setup_logging
from ..logging.logging_utils import log_performance, log_context, structured_log
from .utils import (
    get_start_date_and_seasons,
    validate_data,
    concatenate_scraped_data,
)
from .di_container import DIContainer
from ..error_handling.custom_exceptions import (
    ConfigurationError,
    ScrapingError,
    DataValidationError,
    DataStorageError,
    DataProcessingError,
    WebDriverError,
)

LOG_FILE = "webscraping.log"

@log_performance
def main() -> None:
    """
    Runs the main functionality of the web scraping application.
    
    This function performs the following tasks:
    1. Initializes logging and configuration
    2. Retrieves the start date and seasons to scrape data for
    3. Initializes an NbaScraper instance and uses it to:
        a. Scrape and save all box scores for the specified seasons
        b. Scrape and save the matchups for today's date
    4. Validates the scraped data
    5. Concatenates newly scraped data with cumulative scraped data
    
    The function uses structured logging throughout and implements
    granular error handling for different types of exceptions.
    It also utilizes dependency injection for better modularity.
    """

    container = DIContainer()
    config = container.config()
    data_access = container.data_access()
    nba_scraper = container.nba_scraper()
    web_driver = container.web_driver_factory()
    
    try:
        error_logger = setup_logging(config, LOG_FILE)
        logger = logging.getLogger(__name__)

        structured_log(logger, logging.INFO, "Starting web scraping process", 
                       app_version=config.app_version, 
                       environment=config.environment,
                       log_level=config.log_level,
                       config_summary=str(config.__dict__))

        with log_context(app_version=config.app_version, environment=config.environment):
            first_start_date, seasons = get_scraping_parameters(config, data_access, logger)
            scrape_boxscores(nba_scraper, seasons, first_start_date, logger)
            scrape_matchups(nba_scraper, logger)
            validate_and_concatenate_data(config, data_access, logger)

        structured_log(logger, logging.INFO, "Web scraping process completed successfully")

    except (ConfigurationError, ScrapingError, DataValidationError, 
            DataStorageError, DataProcessingError) as e:
        _handle_known_error(error_logger, e)
    except Exception as e:
        _handle_unexpected_error(error_logger, e)
    finally:
        _close_web_driver(web_driver, error_logger)

def get_scraping_parameters(config, data_access, logger):
    structured_log(logger, logging.INFO, "Retrieving start date and seasons")
    return get_start_date_and_seasons(config, data_access)

def scrape_boxscores(nba_scraper, seasons, first_start_date, logger):
    structured_log(logger, logging.INFO, "Initiating boxscore scraping", 
                   start_date=first_start_date, 
                   seasons=seasons)
    nba_scraper.scrape_and_save_all_boxscores(seasons, first_start_date)
    structured_log(logger, logging.INFO, "Boxscore scraping completed")

def scrape_matchups(nba_scraper, logger):
    search_day = datetime.today().strftime('%A, %B %d')[:3]
    structured_log(logger, logging.INFO, "Initiating matchup scraping", 
                   search_day=search_day)
    if nba_scraper.scrape_and_save_matchups_for_day(search_day):
        structured_log(logger, logging.INFO, "Matchup scraping completed")
    else:
        structured_log(logger, logging.INFO, "No matchups found for today")

def validate_and_concatenate_data(config, data_access, logger):
    structured_log(logger, logging.INFO, "Validating newly scraped data")
    validate_data(config, data_access, cumulative=False)
    structured_log(logger, logging.INFO, "New data validation completed")

    structured_log(logger, logging.INFO, "Concatenating newly scraped data with cumulative data")
    concatenate_scraped_data(config, data_access)
    structured_log(logger, logging.INFO, "Data concatenation completed")

    structured_log(logger, logging.INFO, "Validating cumulative scraped data")
    validate_data(config, data_access, cumulative=True)
    structured_log(logger, logging.INFO, "Cumulative data validation completed")

def _handle_known_error(error_logger, e):
    structured_log(error_logger, logging.ERROR, f"{type(e).__name__} occurred", 
                   error_message=str(e),
                   error_type=type(e).__name__,
                   traceback=traceback.format_exc())
    sys.exit(type(e).exit_code)

def _handle_unexpected_error(error_logger, e):
    structured_log(error_logger, logging.CRITICAL, "Unexpected error occurred", 
                   error_message=str(e),
                   error_type=type(e).__name__,
                   traceback=traceback.format_exc())
    sys.exit(6)

def _close_web_driver(web_driver, error_logger):
    try:
        web_driver.close_driver()
    except WebDriverError as e:
        structured_log(error_logger, logging.ERROR, "Error closing WebDriver", 
                       error_message=str(e),
                       error_type=type(e).__name__)

if __name__ == "__main__":
    main()