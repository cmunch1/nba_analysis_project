import sys
import traceback
import logging

from ..logging.logging_setup import setup_logging
from ..logging.logging_utils import log_performance, log_context, structured_log

from .di_container import DIContainer
from ..error_handling.custom_exceptions import (
    ConfigurationError,
    DataValidationError,
    DataStorageError,
    FeatureEngineeringError,
)

LOG_FILE = "feature_engineering.log"

@log_performance
def main() -> None:
    """
    Main function to perform feature engineering on processed NBA data.
    """

    container = DIContainer()
    config = container.config()
    data_access = container.data_access()
    data_validator = container.data_validator()
    feature_engineer = container.feature_engineer()
    feature_selector = container.feature_selector()

    try:
        error_logger = setup_logging(config, LOG_FILE)
        logger = logging.getLogger(__name__)

        structured_log(logger, logging.INFO, "Starting feature engineering", 
                       app_version=config.app_version, 
                       environment=config.environment,
                       log_level=config.log_level,
                       config_summary=str(config.__dict__))

        with log_context(app_version=config.app_version, environment=config.environment):
            
            processed_dataframe = data_access.load_dataframe(config.team_centric_data_file)

            if not data_validator.validate_processed_dataframe(processed_dataframe, config.team_centric_data_file):

                raise DataValidationError("Data validation of processed data failed")

            # engineer features
            engineered_dataframe = feature_engineer.engineer_features(processed_dataframe)

            # merge home and away team data for each game into a single row
            engineered_dataframe = feature_engineer.merge_team_data(engineered_dataframe)

            # encode game date
            engineered_dataframe = feature_engineer.encode_game_date(engineered_dataframe)
            
            # save engineered dataframe
            data_access.save_dataframes([engineered_dataframe], [config.engineered_data_file])

            # select features
            training_dataframe = feature_selector.select_features(engineered_dataframe)

            # split into training and validation sets
            training_dataframe, validation_dataframe = feature_selector.split_data(training_dataframe)

            # save selected features dataframes
            data_access.save_dataframes([training_dataframe, validation_dataframe], 
                                        [config.training_data_file, config.validation_data_file])


        structured_log(logger, logging.INFO, "Feature engineering completed successfully")

    except (ConfigurationError, DataValidationError, 
            DataStorageError, FeatureEngineeringError) as e:
        _handle_known_error(error_logger, e)
    except Exception as e:
        _handle_unexpected_error(error_logger, e)

def _handle_known_error(error_logger, e):
    structured_log(error_logger, logging.ERROR, f"{type(e).__name__} occurred", 
                   error_message=str(e),
                   error_type=type(e).__name__,
                   traceback=traceback.format_exc())
    sys.exit(1)

def _handle_unexpected_error(error_logger, e):
    structured_log(error_logger, logging.CRITICAL, "Unexpected error occurred", 
                   error_message=str(e),
                   error_type=type(e).__name__,
                   traceback=traceback.format_exc())
    sys.exit(6)

if __name__ == "__main__":
    main()