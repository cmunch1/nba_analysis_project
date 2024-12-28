from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Any, Dict, Union, List
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from ..config.config import AbstractConfig

class AbstractModelTester(ABC):
    @abstractmethod
    def __init__(self, config: AbstractConfig):
        pass

    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        pass

    @abstractmethod
    def perform_oof_cross_validation(self, X: pd.DataFrame, y: pd.Series, model_name: str, model: Union[XGBClassifier, LGBMClassifier, RandomForestClassifier], cv_type: str, n_splits: int) -> Dict[str, np.ndarray]:
        pass
        
    @abstractmethod
    def calculate_classification_evaluation_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        pass

    @abstractmethod
    def get_model_params(self, model_name: str) -> Tuple[Any, Dict]:
        """
        Abstract method to get the current model parameters.
        """
        pass

class Preprocessor:    

    @abstractmethod
    def fit_transform(self, X: np.ndarray, numerical_features: List[str], categorical_features: List[str]) -> np.ndarray:
        pass

class AbstractExperimentLogger(ABC):
    @abstractmethod
    def __init__(self, config: AbstractConfig):
        pass

    @abstractmethod
    def log_experiment(self, experiment_name: str, experiment_description: str, model_name: str, 
                      model: object, model_params: dict, oof_metrics: dict, validation_metrics: dict, 
                      oof_data: pd.DataFrame, val_data: pd.DataFrame):
        pass

    @abstractmethod
    def log_model(self, model: object, model_name: str, model_params: dict):
        pass
