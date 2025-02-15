from abc import ABC, abstractmethod
from typing import Dict, Tuple
import pandas as pd
from ...common.data_classes import ModelTrainingResults
from ...common.config_management.base_config_manager import BaseConfigManager
from ...common.app_logging.base_app_logger import BaseAppLogger
from ...common.error_handling.base_error_handler import BaseErrorHandler
import logging

class BaseTrainer(ABC):
    def __init__(self, config: BaseConfigManager, app_logger: BaseAppLogger, error_handler: BaseErrorHandler):
        """Initialize trainer with configuration and logging."""
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler
        self.app_logger.structured_log(logging.INFO, "BaseTrainer initialized successfully",
                                     trainer_type=type(self).__name__)

    @property
    def log_performance(self):
        return self.app_logger.log_performance

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame, y_val: pd.Series, 
              fold: int, model_params: Dict,
              results: ModelTrainingResults) -> ModelTrainingResults:
        """Train a model and return results"""
        pass

    @abstractmethod
    def _calculate_feature_importance(self, model, X_train, results: ModelTrainingResults) -> None:
        """Calculate and store feature importance scores."""
        pass

    def _convert_metric_scores(self, train_score: float, val_score: float, metric_name: str) -> Tuple[float, float]:
        """Convert metric scores to a consistent format (higher is better)."""
        lower_is_better = ['logloss', 'binary_logloss', 'multi_logloss', 'rmse', 'mae']
        
        if any(metric in metric_name.lower() for metric in lower_is_better):
            return -train_score, -val_score
        
        return train_score, val_score 