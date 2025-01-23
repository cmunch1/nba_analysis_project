from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Any, Dict, Union, List, Optional, Callable
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

class AbstractHyperparameterManager(ABC):
    @abstractmethod
    def get_current_params(self, model_name: str) -> Dict[str, Any]:
        pass    
    
    @abstractmethod
    def update_best_params(self, model_name: str, new_params: Dict[str, Any], metrics: Dict[str, float], experiment_id: str, run_id: str, description: Optional[str] = None) -> None:
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

class AbstractHyperparameterOptimizer(ABC):
    """Abstract base class for hyperparameter optimizers."""
    
    @abstractmethod
    def __init__(self, config: AbstractConfig):
        pass
      
    @abstractmethod
    def optimize(self, 
                objective_func: Optional[Callable] = None,
                param_space: Dict[str, Any] = None,
                X = None,
                y = None,
                model_type: str = None,
                cv: int = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters using the specified objective function.
        
        Args:
            objective_func: Optional custom objective function
            param_space: Dictionary defining parameter search space
            X: Training features
            y: Training labels
            model_type: Type of model to optimize
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary containing best parameters and optimization results
        """
        pass
    
    @abstractmethod
    def get_best_params(self) -> Dict[str, Any]:
        """Return the best parameters found during optimization."""
        pass