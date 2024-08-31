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
    """

    # Initialize the Dependency Injection container
    container = DIContainer()

    # Get instances from the container
    config = container.config()
    data_access = container.data_access()
    nba_scraper = container.nba_scraper()
    web_driver = container.web_driver_factory() 
    
    try:
        # Setup logging
        error_logger = setup_logging(config, LOG_FILE)
        logger = logging.getLogger(__name__)

        # Log start of process with configuration details
        structured_log(logger, logging.INFO, "Starting web scraping process", 
                       app_version=config.app_version, 
                       environment=config.environment,
                       log_level=config.log_level,
                       config_summary=str(config.__dict__))

        # Use log_context to add app_version and environment to all logs
        with log_context(app_version=config.app_version, environment=config.environment):
            # Determine start date and seasons to scrape data for
            structured_log(logger, logging.INFO, "Retrieving start date and seasons")
            #first_start_date, seasons = get_start_date_and_seasons(config, data_access)

            first_start_date = "05/11/2024"
            seasons = ["2023-24"]
            
            # Scrape Boxscores
            structured_log(logger, logging.INFO, "Initiating boxscore scraping", 
                           start_date=first_start_date, 
                           seasons=seasons)
            nba_scraper.scrape_and_save_all_boxscores(seasons, first_start_date)
            structured_log(logger, logging.INFO, "Boxscore scraping completed")

            # Scrape Schedule for Matchups
            search_day = datetime.today().strftime('%A, %B %d')[:3]
            structured_log(logger, logging.INFO, "Initiating matchup scraping", 
                           search_day=search_day)
            if nba_scraper.scrape_and_save_matchups_for_day(search_day):
                structured_log(logger, logging.INFO, "Matchup scraping completed")
            else:
                structured_log(logger, logging.INFO, "No matchups found for today")
 
            # Validate newly scraped data
            structured_log(logger, logging.INFO, "Validating newly scraped data")
            validate_data(config, data_access, cumulative=False)
            structured_log(logger, logging.INFO, "New data validation completed")

            # Combine newly scraped data with cumulative scraped data
            structured_log(logger, logging.INFO, "Concatenating newly scraped data with cumulative data")
            #concatenate_scraped_data(config, data_access)
            structured_log(logger, logging.INFO, "Data concatenation completed")
        
            # Validate cumulative data
            structured_log(logger, logging.INFO, "Validating cumulative scraped data")
            validate_data(config, data_access, cumulative=True)
            structured_log(logger, logging.INFO, "Cumulative data validation completed")

        structured_log(logger, logging.INFO, "Web scraping process completed successfully")

    except ConfigurationError as e:
        structured_log(error_logger, logging.ERROR, "Configuration error occurred", 
                       error_message=str(e),
                       error_type=type(e).__name__,
                       traceback=traceback.format_exc())
        sys.exit(1)
    except ScrapingError as e:
        structured_log(error_logger, logging.ERROR, "Scraping error occurred", 
                       error_message=str(e),
                       error_type=type(e).__name__,
                       traceback=traceback.format_exc())
        sys.exit(2)
    except DataValidationError as e:
        structured_log(error_logger, logging.ERROR, "Data validation error occurred", 
                       error_message=str(e),
                       error_type=type(e).__name__,
                       traceback=traceback.format_exc())
        sys.exit(3)
    except DataStorageError as e:
        structured_log(error_logger, logging.ERROR, "Data storage error occurred", 
                       error_message=str(e),
                       error_type=type(e).__name__,
                       traceback=traceback.format_exc())
        sys.exit(4)
    except DataProcessingError as e:
        structured_log(error_logger, logging.ERROR, "Data processing error occurred", 
                       error_message=str(e),
                       error_type=type(e).__name__,
                       traceback=traceback.format_exc())
        sys.exit(5)
    except Exception as e:
        structured_log(error_logger, logging.CRITICAL, "Unexpected error occurred", 
                       error_message=str(e),
                       error_type=type(e).__name__,
                       traceback=traceback.format_exc())
        sys.exit(6)

    finally:
        # Close the WebDriver
        try:
            web_driver.close_driver()
        except WebDriverError as e:
            structured_log(error_logger, logging.ERROR, "Error closing WebDriver", 
                           error_message=str(e),
                           error_type=type(e).__name__)

if __name__ == "__main__":
    main()