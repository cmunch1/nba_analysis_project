from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import pandas as pd
from src.common.framework.data_classes import ModelTrainingResults
from src.common.core.config_management.base_config_manager import BaseConfigManager
from src.common.core.app_logging.base_app_logger import BaseAppLogger
from src.common.core.error_handling.base_error_handler import BaseErrorHandler

class BaseTrainer(ABC):
    """Abstract base class for model trainers."""
    
    @abstractmethod
    def __init__(self, 
                 config: BaseConfigManager, 
                 app_logger: BaseAppLogger, 
                 error_handler: BaseErrorHandler):
        """
        Initialize trainer with configuration and dependencies.
        
        Args:
            config: Configuration manager
            app_logger: Application logger for structured logging
            error_handler: Error handler for standardized error management
        """
        pass

    @property
    @abstractmethod
    def log_performance(self):
        """Get the performance logging decorator from app_logger."""
        pass

    @abstractmethod
    def train(self, 
             X_train: pd.DataFrame, 
             y_train: pd.Series,
             X_val: pd.DataFrame, 
             y_val: pd.Series,
             fold: int, 
             model_params: Dict[str, Any],
             results: ModelTrainingResults) -> ModelTrainingResults:
        """
        Train a model and return results.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            fold: Current fold number
            model_params: Model parameters
            results: ModelTrainingResults object to store results
            
        Returns:
            Updated ModelTrainingResults object
        """
        pass

    @abstractmethod
    def _calculate_feature_importance(self, 
                                    model: Any, 
                                    X_train: pd.DataFrame,
                                    results: ModelTrainingResults) -> None:
        """
        Calculate and store feature importance scores.
        
        Args:
            model: Trained model object
            X_train: Training features
            results: ModelTrainingResults object to update with importance scores
        """
        pass

    @abstractmethod
    def _convert_metric_scores(self, 
                             train_score: float, 
                             val_score: float, 
                             metric_name: str) -> Tuple[float, float]:
        """
        Convert metric scores to a consistent format (higher is better).
        
        Args:
            train_score: Training metric score
            val_score: Validation metric score
            metric_name: Name of the metric
            
        Returns:
            Tuple of (converted_train_score, converted_val_score)
        """
        pass

    @abstractmethod
    def _process_learning_curve_data(self, 
                                   evals_result: Dict, 
                                   results: ModelTrainingResults) -> None:
        """
        Process and store learning curve data.
        
        Args:
            evals_result: Dictionary containing evaluation metrics
            results: ModelTrainingResults object to update
        """
        pass