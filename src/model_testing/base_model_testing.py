from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple, Any, Dict, List
from ..common.core.config_management.base_config_manager import BaseConfigManager
from ..common.core.app_logging.base_app_logger import BaseAppLogger
from ..common.core.error_handling.base_error_handler import BaseErrorHandler
from ..common.framework.data_classes import ModelTrainingResults, PreprocessingResults

class BaseModelTester(ABC):
    @abstractmethod
    def __init__(self, 
                 config: BaseConfigManager,
                 hyperparameter_manager: Any,
                 trainers: Dict[str, Any],
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        """
        Initialize the model tester with required dependencies.
        
        Args:
            config: Configuration manager
            hyperparameter_manager: Manager for model hyperparameters
            trainers: Dictionary mapping model names to their trainers
            app_logger: Application logger
            error_handler: Error handling utility
        """
        pass
        
    @abstractmethod
    def prepare_data(self, 
                    df: pd.DataFrame, 
                    model_name: str = None,
                    is_training: bool = True,
                    preprocessing_results: PreprocessingResults = None) -> Tuple[pd.DataFrame, pd.Series, PreprocessingResults, pd.Series]:
        """
        Prepare data for model training or validation.
        
        Args:
            df: Input dataframe
            model_name: Name of the model
            is_training: Whether this is training data
            preprocessing_results: Optional existing preprocessing results
            
        Returns:
            Tuple containing:
            - Feature dataframe
            - Target series
            - Preprocessing results
            - Primary IDs series
        """
        pass

    @abstractmethod
    def perform_oof_cross_validation(self, 
                                   X: pd.DataFrame, 
                                   y: pd.Series,
                                   model_name: str,
                                   model_params: Dict,
                                   full_results: ModelTrainingResults) -> ModelTrainingResults:
        """
        Perform Out-of-Fold cross-validation.
        
        Args:
            X: Feature dataframe
            y: Target series
            model_name: Name of the model to use
            model_params: Model parameters
            full_results: ModelTrainingResults object to store results
            
        Returns:
            Updated ModelTrainingResults object
        """
        pass

    @property
    @abstractmethod
    def log_performance(self):
        """Get the performance logging decorator from app_logger."""
        pass

    @abstractmethod
    def get_model_config(self, model_name: str) -> Any:
        """Get model-specific configuration."""
        pass    

    @abstractmethod
    def get_model_config_value(self, model_name: str, key: str, default: Any) -> Any:
        """Get a configuration value with fallback to default."""
        pass

    





