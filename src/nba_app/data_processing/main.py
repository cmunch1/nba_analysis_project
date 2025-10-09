"""main.py

This module serves as the main entry point for the NBA data processing application.
It orchestrates the process of loading scraped NBA data, processing it, validating it,
and saving the processed results.

Key features:
- Loads and validates scraped NBA data
- Processes and transforms the data
- Creates both team-centric and game-centric datasets
- Creates and saves a column mapping file
- Saves invalid records for further review
"""

import sys
import traceback
import logging

from .di_container import DIContainer

LOG_FILE = "data_processing.log"

def main() -> None:
    """
    Main function to process scraped NBA data including cleaning the data and combining it into a single dataframe.
    """

    container = DIContainer()
    app_logger = None

    try:
        config = container.config()

        # Setup the app logger (will use log_path from config)
        app_logger = container.app_logger()
        app_logger.setup(LOG_FILE)

        data_access = container.data_access()
        data_validator = container.data_validator()
        process_scraped_NBA_data = container.process_scraped_NBA_data()
        error_handler = container.error_handler_factory()

        app_logger.structured_log(
            logging.INFO,
            "Starting data processing",
            app_version=config.app_version,
            environment=config.environment,
            log_level=config.core.app_logging_config.log_level if hasattr(config, 'core') else 'INFO',
            config_summary=str(config.__dict__)
        )

        with app_logger.log_context(app_version=config.app_version, environment=config.environment):

            scraped_dataframes, file_names = data_access.load_scraped_data(cumulative=True)

            app_logger.structured_log(logging.INFO, "Validating scraped dataframes")
            if not data_validator.validate_scraped_dataframes(scraped_dataframes, file_names):
                raise error_handler.create_error_handler('data_validation', "Initial data validation of unprocessed scraped data failed")

            app_logger.structured_log(logging.INFO, "Processing scraped data")
            processed_dataframe, column_mapping = process_scraped_NBA_data.process_data(scraped_dataframes)
            processed_file_name = config.team_centric_data_file

            # Validate home/visitor team assignments BEFORE schema validation
            # (uses original_game_id which gets dropped after validation)
            app_logger.structured_log(logging.INFO, "Validating home/visitor team assignments")
            valid_dataframe, invalid_dataframe = process_scraped_NBA_data.validate_home_visitor_teams(processed_dataframe)

            # Continue with only valid records (original_game_id already dropped by validate_home_visitor_teams)
            processed_dataframe = valid_dataframe

            app_logger.structured_log(logging.INFO, "Validating processed dataframe")
            if not data_validator.validate_processed_dataframe(processed_dataframe, processed_file_name):
                raise error_handler.create_error_handler('data_validation', "Data validation of processed data failed")

            app_logger.structured_log(logging.INFO, "Saving team-centric data", file_name=processed_file_name)
            data_access.save_dataframes([processed_dataframe], [processed_file_name], cumulative=True) # expects a list of dataframes and a list of file names
            data_access.save_column_mapping(column_mapping, config.column_mapping_file)

            # combine team data into a single row per game
            app_logger.structured_log(logging.INFO, "Merging team data into game-centric format")
            processed_dataframe = process_scraped_NBA_data.merge_team_data(processed_dataframe)
            processed_file_name = config.game_centric_data_file

            app_logger.structured_log(logging.INFO, "Saving game-centric data", file_name=processed_file_name)
            data_access.save_dataframes([processed_dataframe], [processed_file_name], cumulative=True) # expects a list of dataframes and a list of file names

            # Save invalid records if any exist (team-centric only for manual fixing)
            if len(invalid_dataframe) > 0:
                app_logger.structured_log(logging.WARNING, "Saving invalid home/visitor records for manual review",
                                       invalid_count=len(invalid_dataframe))
                process_scraped_NBA_data.save_invalid_records(invalid_dataframe)

        app_logger.structured_log(logging.INFO, "Data processing completed successfully")

    except Exception as e:
        # Check if it's one of our custom error types (has app_logger and exit_code)
        if hasattr(e, 'app_logger') and hasattr(e, 'exit_code'):
            _handle_known_error(app_logger, e)
        else:
            _handle_unexpected_error(app_logger, e)


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


if __name__ == "__main__":
    main()