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
import pandas as pd

# Using standard logging configuration
from platform_core.core.app_logging import log_performance, log_context, structured_log
from .utils import (
    get_start_date_and_seasons,
)
from .di_container import DIContainer
from platform_core.core.error_handling.error_handler import (
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
    
    # Configure basic logging first
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        config = container.config()

        # Setup the app logger
        app_logger = container.app_logger()
        app_logger.setup(LOG_FILE)

        data_access = container.data_access()
        nba_scraper = container.nba_scraper()
        web_driver = container.web_driver_factory()
        data_validator = container.data_validator()

        structured_log(logger, logging.INFO, "Starting web scraping process", 
                       app_version=config.app_version, 
                       environment=config.environment,
                       log_level=config.log_level,
                       config_summary=str(config.__dict__))

        with log_context(app_version=config.app_version, environment=config.environment):
            
            first_start_date, seasons = get_start_date_and_seasons(config, data_access)
            scrape_boxscores(nba_scraper, seasons, first_start_date, logger)
            scrape_matchups(nba_scraper, logger)
            
            newly_scraped, file_names = data_access.load_scraped_data(cumulative=False)
            
            if config.full_scrape:
                concatenated_data = newly_scraped #no need to concatenate if full scrape is true
            else:        
                cumulative_scraped, file_names = data_access.load_scraped_data(cumulative=True)
                if validate_data(newly_scraped, cumulative_scraped, file_names, data_validator, logger):
                    concatenated_data = concatenate_scraped_data(config, newly_scraped, cumulative_scraped, logger)
                    
            data_access.save_dataframes(concatenated_data, file_names, cumulative=True)

        structured_log(logger, logging.INFO, "Web scraping process completed successfully")

    except (ConfigurationError, ScrapingError, DataValidationError, 
            DataStorageError, DataProcessingError) as e:
        _handle_known_error(logger, e)
    except Exception as e:
        _handle_unexpected_error(logger, e)
    finally:
        _close_web_driver(container.web_driver_factory() if 'container' in locals() else None, logger)


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

def validate_data(newly_scraped, cumulative_scraped, file_names, data_validator, logger) -> bool:

    if not newly_scraped or not cumulative_scraped:
        raise DataValidationError("Either newly scraped or cumulative data is missing")

    structured_log(logger, logging.INFO, "Validating newly scraped data")
    if not data_validator.validate_scraped_dataframes(newly_scraped, file_names):
        raise DataValidationError("Data validation failed")
    
    structured_log(logger, logging.INFO, "Validating cumulative scraped data")
    if not data_validator.validate_scraped_dataframes(cumulative_scraped, file_names):
        raise DataValidationError("Data validation failed")

    structured_log(logger, logging.INFO, "Data validation completed")

    return True

def concatenate_scraped_data(config, newly_scraped, cumulative_scraped, logger) -> list[pd.DataFrame]:
    """
    Concatenate newly scraped data with cumulative scraped data.

    Args:
        config (BaseConfigManager): The configuration object.
        data_access (BaseDataAccess): The data access object.
        logger (logging.Logger): The logger object for structured logging.

    Raises:
        DataValidationError: If there are issues with the scraped data.
        DataProcessingError: If there's an error during the concatenation process.
    """
    try:
        structured_log(logger, logging.INFO, "Starting data concatenation process")
        combined_dataframes = []

        for i, (new_df, cum_df, file_name) in enumerate(zip(newly_scraped, cumulative_scraped, config.scraped_boxscore_files)):
            if new_df.empty:
                structured_log(logger, logging.WARNING, "Skipping empty dataframe", file_name=file_name)
                continue

            combined_df = pd.concat([cum_df, new_df], ignore_index=True)
            combined_df = combined_df.sort_values(by=[config.game_id_column, config.team_id_column])
            combined_df = combined_df.drop_duplicates(subset=[config.game_id_column, config.team_id_column], keep='last')
            
            combined_dataframes.append(combined_df)
            
            structured_log(logger, logging.INFO, "Successfully concatenated and saved file", file_name=file_name)

        structured_log(logger, logging.INFO, "Completed concatenation of all scraped data files")

        return combined_dataframes
    except DataValidationError:
        raise
    except Exception as e:
        raise DataProcessingError(f"Error in concatenate_scraped_data: {str(e)}")

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
    if web_driver is None:
        return
    try:
        web_driver.close_driver()
    except WebDriverError as e:
        structured_log(error_logger, logging.ERROR, "Error closing WebDriver", 
                       error_message=str(e),
                       error_type=type(e).__name__)

if __name__ == "__main__":
    main()