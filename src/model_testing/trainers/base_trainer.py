from abc import ABC, abstractmethod
from typing import Dict, Tuple
import pandas as pd
from ...common.data_classes.data_classes import ModelTrainingResults
import logging
from ...config.config import AbstractConfig

logger = logging.getLogger(__name__)

class BaseTrainer(ABC):
    def __init__(self, config: AbstractConfig):
        """Initialize trainer with configuration."""
        self.config = config

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