import pandas as pd
import numpy as np
from typing import List
import logging

from ..config.abstract_config import AbstractConfig
from ..data_access.abstract_data_access import AbstractDataAccess
from ..error_handling.custom_exceptions import DataValidationError, DataProcessingError
from ..logging.logging_utils import log_performance, structured_log
from .abstract_data_validator import AbstractDataValidator

logger = logging.getLogger(__name__)

class DataValidator(AbstractDataValidator):
    @log_performance
    def __init__(self, config: AbstractConfig,):
        super().__init__(config)
        self.config = config
        structured_log(logger, logging.INFO, "DataValidator initialized successfully",
                       config_type=type(config).__name__,
                       )


    @log_performance
    def validate_scraped_dataframes(self, scraped_dataframes: List[pd.DataFrame], file_names: List[str]) -> bool:
        """
        Validate the consistency of scraped dataframes.

        Args:
            scraped_dataframes (List[pd.DataFrame]): List of scraped dataframes to validate.

        Returns:
            bool: True if all dataframes are consistent, otherwise False.

        Raises:
            DataValidationError: If there are inconsistencies in the scraped dataframes.
        """
        try:
            structured_log(logger, logging.INFO, "Starting validation of scraped dataframes",
                           dataframe_count=len(scraped_dataframes))
            num_rows = 0
            game_ids = None
        
            for i, df in enumerate(scraped_dataframes):
                if df.duplicated().any():
                    raise DataValidationError(f"Dataframe {file_names[i]} has duplicate records")
     
                if df.isnull().values.any():
                    raise DataValidationError(f"Dataframe {file_names[i]} has null values")
                
                df = df.sort_values(by=self.config.game_id_column)

                if i == 0:
                    num_rows = df.shape[0]
                    game_ids = df[self.config.game_id_column]
                else:
                    if num_rows != df.shape[0]:
                        raise DataValidationError(f"Dataframe {file_names[i]} does not match the number of rows of the first dataframe")
                    
                    if not np.array_equal(game_ids.values, df[self.config.game_id_column].values):
                        raise DataValidationError(f"Dataframe {file_names[i]} does not match the game ids of the first dataframe")
                
                structured_log(logger, logging.INFO, f"Dataframe {file_names[i]} validated successfully",
                               rows=df.shape[0], columns=df.shape[1])
            
            structured_log(logger, logging.INFO, "All dataframes validated successfully")
            return True
        except DataValidationError:
            raise
        except Exception as e:
            raise DataProcessingError(f"Error in validate_scraped_dataframes: {str(e)}")