import pandas as pd
import numpy as np
from typing import List
import logging

from platform_core.core.config_management.base_config_manager import BaseConfigManager
from platform_core.core.app_logging.base_app_logger import BaseAppLogger
from platform_core.framework.data_access.base_data_access import BaseDataAccess
from platform_core.core.app_file_handling.base_app_file_handler import BaseAppFileHandler
from platform_core.core.error_handling.base_error_handler import BaseErrorHandler
from platform_core.framework.base_data_validator import BaseDataValidator


class DataValidator(BaseDataValidator):
  
    def __init__(self, config: BaseConfigManager, data_access: BaseDataAccess, 
                 app_logger: BaseAppLogger, app_file_handler: BaseAppFileHandler, 
                 error_handler: BaseErrorHandler):
        """
        Initialize the DataValidator with required dependencies.
        
        Args:
            config: Configuration manager instance
            data_access: Data access layer instance
            app_logger: Application logger instance
            app_file_handler: Application file handler instance
            error_handler: Error handler instance
        """
        super().__init__(config, data_access, app_logger, app_file_handler, error_handler)
        self.config = config
        self.data_access = data_access
        self.app_logger = app_logger
        self.app_file_handler = app_file_handler
        self.error_handler = error_handler

    @staticmethod
    def log_performance(func):
        """Decorator factory for performance logging"""
        def wrapper(*args, **kwargs):
            # Get the self instance from args since this is now a static method
            instance = args[0]
            return instance.app_logger.log_performance(func)(*args, **kwargs)
        return wrapper

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
            self.app_logger.structured_log(logging.INFO, "Starting validation of scraped dataframes",
                           dataframe_count=len(scraped_dataframes))
            num_rows = 0
            game_ids = None

            if scraped_dataframes[0].empty:
                self.app_logger.structured_log(logging.WARNING, "First dataframe is empty, skipping validation")
                return True
         
            for i, df in enumerate(scraped_dataframes):
                
                if not self._validate_dataframe(df, file_names[i]):
                    raise self.error_handler.create_error_handler(
                        'data_validation',
                        f"Dataframe {file_names[i]} failed validation",
                        file_name=file_names[i]
                    )
                
                
                self.app_logger.structured_log(logging.INFO, "Checking that same games are present")
                           
                df = df.sort_values(by=self.config.game_id_column)

                if i == 0:
                    num_rows = df.shape[0]
                    game_ids = df[self.config.game_id_column]
                else:
                    if num_rows != df.shape[0]:
                        raise self.error_handler.create_error_handler(
                            'data_validation',
                            f"Dataframe {file_names[i]} does not match the number of rows of the first dataframe",
                            file_name=file_names[i],
                            expected_rows=num_rows,
                            actual_rows=df.shape[0]
                        )
                    
                    if not np.array_equal(game_ids.values, df[self.config.game_id_column].values):
                        raise self.error_handler.create_error_handler(
                            'data_validation',
                            f"Dataframe {file_names[i]} does not match the game ids of the first dataframe",
                            file_name=file_names[i]
                        )
                
                self.app_logger.structured_log(logging.INFO, f"Dataframe {file_names[i]} validated successfully",
                               rows=df.shape[0], columns=df.shape[1])
            
            self.app_logger.structured_log(logging.INFO, "All dataframes validated successfully")
            return True
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_processing',
                f"Error in validate_scraped_dataframes: {str(e)}",
                original_error=str(e)
            )
        
    @log_performance
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
            self.app_logger.structured_log(logging.INFO, f"Starting validation of processed dataframe {file_name}")

            if not self._validate_dataframe(df, file_name):
                raise self.error_handler.create_error_handler(
                    'data_validation',
                    f"Dataframe {file_name} failed validation",
                    file_name=file_name
                )
            
            if not self._validate_schema(df, file_name, self.config.processed_schema):
                raise self.error_handler.create_error_handler(
                    'data_validation',
                    f"Dataframe {file_name} failed schema validation",
                    file_name=file_name
                )
            
            self.app_logger.structured_log(logging.INFO, f"Dataframe {file_name} validated successfully",
                           rows=df.shape[0], columns=df.shape[1])
            
            return True
        
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_processing',
                f"Error in processed dataframe validation: {str(e)}",
                original_error=str(e)
            )
        
           
    def _validate_dataframe(self, df: pd.DataFrame, file_name: str) -> bool:
        """
        Validate the dataframe for common issues like nulls and duplicates
        
        Args:
            df: dataframe to validate.
            file_name: name of the file being validated.

        Returns:
            bool: True if passes all the checks, False otherwise
        """
        self.app_logger.structured_log(logging.INFO, f"Checking {file_name} for nulls and duplicates")

        if df.duplicated().any():
            self.app_logger.structured_log(logging.WARNING, f"Dataframe {file_name} has duplicate records")
            return False

        if df.isnull().values.any():
            self.app_logger.structured_log(logging.WARNING, f"Dataframe {file_name} has null values")
            return False
        
        self.app_logger.structured_log(logging.INFO, f"Dataframe {file_name} passes initial checks",
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
        self.app_logger.structured_log(logging.INFO, f"Checking {file_name} for correct schema")

        if df.columns.tolist() != schema:
            self.app_logger.structured_log(logging.WARNING, f"Dataframe {file_name} has incorrect schema")
            schema_diff = set(df.columns.tolist()) - set(schema)
            print(df.columns.tolist())
            print(schema)
            self.app_logger.structured_log(logging.WARNING, f"Columns in dataframe that are not in schema: {schema_diff}")
            return False    
        
        self.app_logger.structured_log(logging.INFO, f"Dataframe {file_name} passes schema checks",
                       columns=df.columns.tolist())
        return True 
            
            

