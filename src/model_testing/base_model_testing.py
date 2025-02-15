from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple, Any, Dict, Union, List, Optional, Callable
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from ..common.config_management.base_config_manager import BaseConfigManager

class BaseModelTester(ABC):
    @abstractmethod
    def __init__(self, config: BaseConfigManager):
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


    





