
import sys
import traceback
import logging

from ..logging.logging_setup import setup_logging
from ..logging.logging_utils import log_performance, log_context, structured_log

from .di_container import DIContainer
from ..error_handling.custom_exceptions import (
    ConfigurationError,
    ScrapingError,
    DataValidationError,
    DataStorageError,
    DataProcessingError,
)

LOG_FILE = "data_processing.log"

@log_performance
def main() -> None:
    """
    Main function to process scraped NBA data including cleaning the data and combining it into a single dataframe.
    """

    container = DIContainer()
    config = container.config()
    data_access = container.data_access()
    data_validator = container.data_validator()
    process_scraped_NBA_data = container.process_scraped_NBA_data()

    
    try:
        error_logger = setup_logging(config, LOG_FILE)
        logger = logging.getLogger(__name__)

        structured_log(logger, logging.INFO, "Starting data processing", 
                       app_version=config.app_version, 
                       environment=config.environment,
                       log_level=config.log_level,
                       config_summary=str(config.__dict__))

        with log_context(app_version=config.app_version, environment=config.environment):
            
            scraped_dataframes, file_names = data_access.load_scraped_data(cumulative=True)

            if not data_validator.validate_scraped_dataframes(scraped_dataframes, file_names):
                raise DataValidationError("Initial data validation of unprocessed scraped data failed")

            processed_dataframe = process_scraped_NBA_data.process_data(scraped_dataframes)
            processed_file_name = config.cleaned_and_combined_data_file
            
            if not data_validator.validate_processed_dataframe(processed_dataframe, processed_file_name):
                raise DataValidationError("Data validation of processed data failed")

            data_access.save_dataframes([processed_dataframe], [processed_file_name], cumulative=True) # expects a list of dataframes and a list of file names
            
        structured_log(logger, logging.INFO, "Data processing completed successfully")

    except (ConfigurationError, ScrapingError, DataValidationError, 
            DataStorageError, DataProcessingError) as e:
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