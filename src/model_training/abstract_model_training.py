from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Any, Dict, Union, List
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from ..config.config import AbstractConfig

class AbstractModelTrainer(ABC):

    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        pass

    @abstractmethod
    def perform_cross_validation(self, X: pd.DataFrame, y: pd.Series, model_name: str, model: Union[XGBClassifier, LGBMClassifier, RandomForestClassifier], cv_type: str, n_splits: int) -> Dict[str, np.ndarray]:
        pass
        
    @abstractmethod
    def evaluate_model(self, model: Union[XGBClassifier, LGBMClassifier, RandomForestClassifier], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]  :
        """
        Abstract method to evaluate the model.
        
        Args:
            model: The trained model object.
            X_test (pd.DataFrame): The test feature dataframe.
            y_test (pd.Series): The test target series.
        
        Returns:
            dict: Dictionary of evaluation metrics.
        """
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
