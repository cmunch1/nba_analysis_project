import pandas as pd
import numpy as np
from typing import List
import logging

from ..config.abstract_config import AbstractConfig
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
                
                if not self._validate_dataframe(df, file_names[i]):
                    raise DataValidationError(f"Dataframe {file_names[i]} failed validation")
                
                
                structured_log(logger, logging.INFO, "Checking that same games are present")
                           
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
        
    def validate_processed_dataframe(self, df: pd.DataFrame, file_name: str) -> bool:
        """
        Validate the processed dataframe to insure it was processed properly and has proper schema
        
        Args:
            df: dataframe to validate.
            file_name: name of the file being validated.

        Returns:    
            bool: True if passes all the checks, False otherwise
        """
        try:
            structured_log(logger, logging.INFO, f"Starting validation of processed dataframe {file_name}")

            if not self._validate_dataframe(df, file_name):
                        raise DataValidationError(f"Dataframe {file_name} failed validation")
            
            if not self._validate_schema(df, file_name, self.config.processed_schema):
                raise DataValidationError(f"Dataframe {file_name} failed schema validation")
            
            structured_log(logger, logging.INFO, f"Dataframe {file_name} validated successfully",
                           rows=df.shape[0], columns=df.shape[1])
            
            return True
        
        except DataValidationError:
            raise
        except Exception as e:
            raise DataProcessingError(f"Error in processed dataframe validation: {str(e)}")
        
           
    def _validate_dataframe(self, df: pd.DataFrame, file_name: str) -> bool:
        """
        Validate the dataframe for common issues like nulls and duplicates
        
        Args:
            df: dataframe to validate.
            file_name: name of the file being validated.

        Returns:
            bool: True if passes all the checks, False otherwise
        """
        structured_log(logger, logging.INFO, f"Checking {file_name} for nulls and duplicates")

        if df.duplicated().any():
            structured_log(logger, logging.WARNING, f"Dataframe {file_name} has duplicate records")
            return False

        if df.isnull().values.any():
            structured_log(logger, logging.WARNING, f"Dataframe {file_name} has null values")
            return False
        
        structured_log(logger, logging.INFO, f"Dataframe {file_name} passes initial checks",
                       rows=df.shape[0], columns=df.shape[1])
        return True
    
    def _validate_schema(self, df: pd.DataFrame, file_name: str, schema: List[str]) -> bool:
        """
        Validate the dataframe has the correct schema
        
        Args:
            df: dataframe to validate.
            file_name: name of the file being validated.

        Returns:    
            bool: True if passes all the checks, False otherwise
        """
        structured_log(logger, logging.INFO, f"Checking {file_name} for correct schema")

        if not df.columns.tolist() == schema:
            structured_log(logger, logging.WARNING, f"Dataframe {file_name} has incorrect schema")
            return False    
        
        structured_log(logger, logging.INFO, f"Dataframe {file_name} passes schema checks",
                       columns=df.columns.tolist())
        return True 
            
            

