from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Any

@dataclass
class ModelTrainingResults:
    def __init__(self, X_shape: tuple[int, int]):
        self.predictions: NDArray[np.float_] = np.zeros(X_shape[0])
        self.shap_values: NDArray[np.float_] = np.zeros((X_shape[0], X_shape[1]))
        self.shap_interaction_values: NDArray[np.float_] = np.zeros((X_shape[0], X_shape[1]))
        self.feature_names: list[str] = []
        self.feature_importance_scores: NDArray[np.float_] = np.zeros(X_shape[1])
        self.model: Optional[Any] = None


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float 
    f1: float
    auc: float
    optimal_threshold: float