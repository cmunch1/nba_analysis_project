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

from .utils import (
    get_start_date_and_seasons,
)
from .di_container import DIContainer


LOG_FILE = "webscraping.log"

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
    app_logger = None

    try:
        config = container.config()

        # Setup the app logger (will use log_path from config)
        app_logger = container.app_logger()
        app_logger.setup(LOG_FILE)

        data_access = container.data_access()
        nba_scraper = container.nba_scraper()
        web_driver = container.web_driver_factory()
        data_validator = container.data_validator()
        error_handler = container.error_handler_factory()

        app_logger.structured_log(
            logging.INFO,
            "Starting web scraping process",
            app_version=config.app_version,
            environment=config.environment,
            log_level=config.core.app_logging_config.log_level if hasattr(config, 'core') else 'INFO',
            config_summary=str(config.__dict__)
        )

        with app_logger.log_context(app_version=config.app_version, environment=config.environment):

            first_start_date, seasons = get_start_date_and_seasons(config, data_access, error_handler)
            scrape_boxscores(nba_scraper, seasons, first_start_date, app_logger)
            scrape_matchups(nba_scraper, app_logger)

            newly_scraped, file_names = data_access.load_scraped_data(cumulative=False)

            # Scrape validation data if enabled
            if hasattr(config, 'enable_validation_scraping') and config.enable_validation_scraping:
                scrape_validation_data(nba_scraper, newly_scraped, config, app_logger)

            if config.full_scrape:
                concatenated_data = newly_scraped #no need to concatenate if full scrape is true
            else:
                cumulative_scraped, file_names = data_access.load_scraped_data(cumulative=True)
                if validate_data(newly_scraped, cumulative_scraped, file_names, data_validator, error_handler, app_logger):
                    concatenated_data = concatenate_scraped_data(config, newly_scraped, cumulative_scraped, error_handler, app_logger)

            data_access.save_dataframes(concatenated_data, file_names, cumulative=True)

        app_logger.structured_log(logging.INFO, "Web scraping process completed successfully")

    except Exception as e:
        # Check if it's one of our custom error types (has app_logger and exit_code)
        if hasattr(e, 'app_logger') and hasattr(e, 'exit_code'):
            _handle_known_error(app_logger, e)
        else:
            _handle_unexpected_error(app_logger, e)
    finally:
        _close_web_driver(container.web_driver_factory() if 'container' in locals() else None, app_logger)


def scrape_boxscores(nba_scraper, seasons, first_start_date, app_logger):
    app_logger.structured_log(
        logging.INFO,
        "Initiating boxscore scraping",
        start_date=first_start_date,
        seasons=seasons
    )
    nba_scraper.scrape_and_save_all_boxscores(seasons, first_start_date)
    app_logger.structured_log(logging.INFO, "Boxscore scraping completed")

def scrape_matchups(nba_scraper, app_logger):
    search_day = datetime.today().strftime('%A, %B %d')[:3]
    app_logger.structured_log(
        logging.INFO,
        "Initiating matchup scraping",
        search_day=search_day
    )
    if nba_scraper.scrape_and_save_matchups_for_day(search_day):
        app_logger.structured_log(logging.INFO, "Matchup scraping completed")
    else:
        app_logger.structured_log(logging.INFO, "No matchups found for today")

def scrape_validation_data(nba_scraper, newly_scraped, config, app_logger):
    """
    Scrape validation data from basketball-reference.com for all games in newly scraped data.

    Args:
        nba_scraper: NbaScraper instance
        newly_scraped: List of newly scraped dataframes
        config: Configuration object
        app_logger: App logger instance
    """
    try:
        app_logger.structured_log(logging.INFO, "Initiating validation data scraping")

        # Build game metadata from scraped data
        # We need: GAME_ID, GAME_DATE, HOME_TEAM_ID (team with "vs." in matchup)
        game_metadata_list = []

        for df in newly_scraped:
            if df.empty:
                continue

            # Verify required columns exist
            required_cols = [config.game_id_column, 'GAME DATE', 'MATCH UP', 'TEAM_ID']
            if not all(col in df.columns for col in required_cols):
                app_logger.structured_log(
                    logging.WARNING,
                    "Missing required columns for validation scraping",
                    available_columns=list(df.columns)
                )
                continue

            # Filter to only home teams (those with "vs." in matchup)
            home_games = df[df['MATCH UP'].str.contains('vs.', case=False, na=False)].copy()

            if not home_games.empty:
                game_metadata_list.append(home_games[[config.game_id_column, 'GAME DATE', 'TEAM_ID']])

        if not game_metadata_list:
            app_logger.structured_log(
                logging.WARNING,
                "No game metadata extracted from scraped data - skipping validation scraping"
            )
            return

        # Combine and deduplicate game metadata
        game_metadata = pd.concat(game_metadata_list, ignore_index=True)
        game_metadata = game_metadata.drop_duplicates(subset=[config.game_id_column])

        # Rename columns to match ValidationScraper expectations
        game_metadata = game_metadata.rename(columns={
            config.game_id_column: 'GAME_ID',
            'GAME DATE': 'GAME_DATE',
            'TEAM_ID': 'HOME_TEAM_ID'
        })

        app_logger.structured_log(
            logging.INFO,
            "Scraping validation data for games",
            game_count=len(game_metadata)
        )

        if nba_scraper.scrape_and_save_validation_data(game_metadata):
            app_logger.structured_log(logging.INFO, "Validation data scraping completed successfully")
        else:
            app_logger.structured_log(logging.WARNING, "Validation data scraping skipped or failed")

    except Exception as e:
        app_logger.structured_log(
            logging.ERROR,
            "Error during validation data scraping",
            error_message=str(e),
            error_type=type(e).__name__
        )
        # Don't fail the entire pipeline if validation scraping fails
        app_logger.structured_log(logging.WARNING, "Continuing pipeline despite validation scraping error")

def validate_data(newly_scraped, cumulative_scraped, file_names, data_validator, error_handler, app_logger) -> bool:

    if not newly_scraped or not cumulative_scraped:
        raise error_handler.create_error_handler('data_validation', "Either newly scraped or cumulative data is missing")

    app_logger.structured_log(logging.INFO, "Validating newly scraped data")
    if not data_validator.validate_scraped_dataframes(newly_scraped, file_names):
        raise error_handler.create_error_handler('data_validation', "Data validation failed")

    app_logger.structured_log(logging.INFO, "Validating cumulative scraped data")
    if not data_validator.validate_scraped_dataframes(cumulative_scraped, file_names):
        raise error_handler.create_error_handler('data_validation', "Data validation failed")

    app_logger.structured_log(logging.INFO, "Data validation completed")

    return True

def concatenate_scraped_data(config, newly_scraped, cumulative_scraped, error_handler, app_logger) -> list[pd.DataFrame]:
    """
    Concatenate newly scraped data with cumulative scraped data.

    Args:
        config (BaseConfigManager): The configuration object.
        newly_scraped (list): List of newly scraped dataframes.
        cumulative_scraped (list): List of cumulative scraped dataframes.
        error_handler (ErrorHandlerFactory): Error handler factory instance.
        app_logger: The app logger object for structured logging.

    Raises:
        DataValidationError: If there are issues with the scraped data.
        DataProcessingError: If there's an error during the concatenation process.
    """
    try:
        app_logger.structured_log(logging.INFO, "Starting data concatenation process")
        combined_dataframes = []

        for new_df, cum_df, file_name in zip(newly_scraped, cumulative_scraped, config.scraped_boxscore_files):
            if new_df.empty:
                app_logger.structured_log(
                    logging.WARNING,
                    "Skipping empty dataframe",
                    file_name=file_name
                )
                continue

            combined_df = pd.concat([cum_df, new_df], ignore_index=True)
            combined_df = combined_df.sort_values(by=[config.game_id_column, config.team_id_column])
            combined_df = combined_df.drop_duplicates(subset=[config.game_id_column, config.team_id_column], keep='last')

            combined_dataframes.append(combined_df)

            app_logger.structured_log(
                logging.INFO,
                "Successfully concatenated and saved file",
                file_name=file_name
            )

        app_logger.structured_log(logging.INFO, "Completed concatenation of all scraped data files")

        return combined_dataframes
    except Exception as e:
        # Check if it's already one of our error types (has app_logger)
        if hasattr(e, 'app_logger'):
            raise
        raise error_handler.create_error_handler('data_processing', f"Error in concatenate_scraped_data: {str(e)}")

def _handle_known_error(app_logger, e):
    if app_logger:
        app_logger.structured_log(
            logging.ERROR,
            f"{type(e).__name__} occurred",
            error_message=str(e),
            error_type=type(e).__name__,
            traceback=traceback.format_exc()
        )
    sys.exit(type(e).exit_code)

def _handle_unexpected_error(app_logger, e):
    if app_logger:
        app_logger.structured_log(
            logging.CRITICAL,
            "Unexpected error occurred",
            error_message=str(e),
            error_type=type(e).__name__,
            traceback=traceback.format_exc()
        )
    else:
        print(f"CRITICAL: Unexpected error occurred: {str(e)}")
        print(traceback.format_exc())
    sys.exit(6)

def _close_web_driver(web_driver, app_logger):
    if web_driver is None:
        return
    try:
        web_driver.close_driver()
    except Exception as e:
        if app_logger:
            app_logger.structured_log(
                logging.ERROR,
                "Error closing WebDriver",
                error_message=str(e),
                error_type=type(e).__name__
            )

if __name__ == "__main__":
    main()