"""main.py

This module serves as the main entry point for the NBA feature engineering application.
It orchestrates the process of loading processed NBA data, engineering features,
selecting features, and creating training/validation datasets.

Key features:
- Loads and validates processed NBA data
- Engineers features including rolling averages, streaks, and ELO ratings
- Merges home/away team data into game-centric format
- Encodes temporal features from game dates
- Performs feature selection
- Splits data into training and validation sets
- Implements comprehensive logging for better debugging and monitoring
- Utilizes structured logging and context managers for consistent log formatting
- Implements granular error handling with custom exceptions
- Uses dependency injection for better modularity and testability
"""

import sys
import traceback
import logging

from .di_container import DIContainer

LOG_FILE = "feature_engineering.log"

def main() -> None:
    """
    Main function to perform feature engineering on processed NBA data.
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
        feature_engineer = container.feature_engineer()
        feature_selector = container.feature_selector()
        error_handler = container.error_handler_factory()

        app_logger.structured_log(
            logging.INFO,
            "Starting feature engineering",
            app_version=config.app_version,
            environment=config.environment,
            log_level=config.core.app_logging_config.log_level if hasattr(config, 'core') else 'INFO',
            config_summary=str(config.__dict__)
        )

        with app_logger.log_context(app_version=config.app_version, environment=config.environment):

            app_logger.structured_log(logging.INFO, "Loading processed dataframe", file_name=config.team_centric_data_file)
            processed_dataframe = data_access.load_dataframe(config.team_centric_data_file)

            app_logger.structured_log(logging.INFO, "Validating processed dataframe")
            if not data_validator.validate_processed_dataframe(processed_dataframe, config.team_centric_data_file):
                raise error_handler.create_error_handler('data_validation', "Data validation of processed data failed")

            # engineer features
            app_logger.structured_log(logging.INFO, "Engineering features")
            engineered_dataframe = feature_engineer.engineer_features(processed_dataframe)

            # merge home and away team data for each game into a single row
            app_logger.structured_log(logging.INFO, "Merging team data into game-centric format")
            engineered_dataframe = feature_engineer.merge_team_data(engineered_dataframe)

            # encode game date
            app_logger.structured_log(logging.INFO, "Encoding game date features")
            engineered_dataframe = feature_engineer.encode_game_date(engineered_dataframe)

            # save engineered dataframe
            app_logger.structured_log(logging.INFO, "Saving engineered features", file_name=config.engineered_data_file)
            data_access.save_dataframes([engineered_dataframe], [config.engineered_data_file])

            # select features
            app_logger.structured_log(logging.INFO, "Selecting features")
            training_dataframe = feature_selector.select_features(engineered_dataframe)

            # split into training and validation sets
            app_logger.structured_log(logging.INFO, "Splitting data into training and validation sets")
            training_dataframe, validation_dataframe = feature_selector.split_data(training_dataframe)

            # save selected features dataframes
            app_logger.structured_log(logging.INFO, "Saving training and validation datasets")
            data_access.save_dataframes([training_dataframe, validation_dataframe],
                                        [config.training_data_file, config.validation_data_file])

        app_logger.structured_log(logging.INFO, "Feature engineering completed successfully")

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