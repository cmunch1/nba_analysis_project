"""Data class for model training results."""

from dataclasses import dataclass
from typing import Optional, Any, Dict, List
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from .metrics import ClassificationMetrics, LearningCurveData
from .preprocessing import PreprocessingResults

@dataclass
class ModelTrainingResults:
    """Comprehensive container for model training results and evaluation metrics"""
    def __init__(self, X_shape: tuple[int, int]):
        # Model-specific fields
        self.predictions: Optional[NDArray[np.float_]] = None
        self.shap_values: Optional[NDArray[np.float_]] = None
        self.shap_interaction_values: Optional[NDArray[np.float_]] = None
        self.feature_names: List[str] = []
        self.feature_importance_scores: Optional[NDArray[np.float_]] = None
        self.model: Optional[Any] = None
        self.model_name: str = ""
        self.model_params: Dict = {}
        self.num_boost_round: int = 0
        self.early_stopping: int = 0
        self.enable_categorical: bool = False
        self.categorical_features: List[str] = []
        self.metrics: Optional[ClassificationMetrics] = None
        self.eval_metric = None  
        
        # Data fields
        self.feature_data: Optional[pd.DataFrame] = pd.DataFrame(np.zeros(X_shape))
        self.target_data: Optional[pd.Series] = None
        self.binary_predictions: Optional[NDArray[np.int_]] = None
        self.probability_predictions: Optional[NDArray[np.float_]] = None
        
        # Evaluation context
        self.is_validation: bool = False
        self.evaluation_type: str = ""
        
        # Preprocessing results
        self.preprocessing_results: Optional[PreprocessingResults] = None
        self.preprocessing_artifact: Optional[Dict[str, Any]] = None  # Fitted preprocessor for persistence

        # Postprocessing results
        self.calibrated_predictions: Optional[NDArray[np.float_]] = None  # Calibrated probabilities
        self.calibration_artifact: Optional[Any] = None  # Fitted calibrator for persistence
        self.calibration_metrics: Optional[Dict[str, float]] = None  # Calibration quality metrics
        self.calibration_curve_data: Optional[Dict[str, Any]] = None  # For visualization

        self.learning_curve_data = LearningCurveData()
        self.n_folds = 0

        # Feature audit results
        self.feature_audit: Optional[pd.DataFrame] = None  # Feature audit DataFrame

        # Per-fold importance tracking for stability analysis
        self.fold_importances: List[Dict[str, float]] = []  # Model gain importance per fold
        self.fold_shap_importances: List[Dict[str, float]] = []  # Mean absolute SHAP per fold

    def add_learning_curve_point(self, train_size: int, train_score: float, val_score: float, 
                               fold: int, iteration: Optional[int] = None, metric_name: Optional[str] = None):
        """Add a learning curve data point."""
        self.learning_curve_data.add_point(
            train_size=train_size,
            train_score=train_score,
            val_score=val_score,
            fold=fold,
            iteration=iteration,
            metric_name=metric_name
        )

    def update_feature_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Update feature and target data"""
        self.feature_data = X
        self.target_data = y
        self.feature_names = X.columns.tolist()

    def update_predictions(self, probability_predictions: NDArray[np.float64], 
                         threshold: float = 0.5) -> None:
        """Update probability and binary predictions"""
        self.probability_predictions = probability_predictions
        self.binary_predictions = (probability_predictions >= threshold).astype(int)
        self.predictions = probability_predictions  # For backward compatibility

    def prepare_for_logging(self) -> Dict[str, Any]:
        """Prepare results for experiment logging"""
        prefix = "val_" if self.is_validation else "oof_"
        
        # Prepare full dataframe with predictions
        if self.feature_data is not None and self.target_data is not None:
            full_data = self.feature_data.copy()
            full_data['target'] = self.target_data
            if self.probability_predictions is not None:
                full_data[f'{prefix}predictions'] = self.probability_predictions
        else:
            full_data = None

        return {
            f"{prefix}data": full_data,
            f"{prefix}metrics": self.metrics,
            "model_name": self.model_name,
            "model": self.model,
            "model_params": self.model_params,
            "preprocessing_results": self.preprocessing_results.to_dict() if self.preprocessing_results else None
        }

    def prepare_for_charting(self) -> Dict[str, Any]:
        """Prepare data for chart generation"""
        return {
            "feature_importance": self.feature_importance_scores,
            "feature_names": self.feature_names,
            "y_true": self.target_data,
            "y_pred": self.binary_predictions,
            "y_prob": self.probability_predictions,
            "model": self.model,
            "X": self.feature_data,
            "prefix": "val_" if self.is_validation else "oof_"
        }
