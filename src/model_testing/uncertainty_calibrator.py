import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from ..logging.logging_utils import log_performance, structured_log
from ..error_handling.custom_exceptions import ModelTestingError
from ..config.config import AbstractConfig
import lightgbm as lgb
import catboost as cb

logger = logging.getLogger(__name__)

class BoostingModelWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper class to make boosting models compatible with sklearn's calibration methods."""
    
    def __init__(self, model):
        self.model = model
        
    def fit(self, X, y):
        return self
    
    def predict_proba(self, X):
        if isinstance(self.model, xgb.Booster):
            dmatrix = xgb.DMatrix(X)
            probas = self.model.predict(dmatrix)
        elif isinstance(self.model, lgb.Booster):
            probas = self.model.predict(X)
        elif isinstance(self.model, cb.CatBoost):
            probas = self.model.predict_proba(X)[:, 1]
        else:
            raise ValueError("Unsupported model type. Must be XGBoost, LightGBM, or CatBoost.")
        
        return np.vstack((1 - probas, probas)).T

class UncertaintyCalibrator:
    @log_performance
    def __init__(self, config: AbstractConfig):
        """
        Initialize the UncertaintyCalibrator class.

        Args:
            config (AbstractConfig): Configuration object containing calibration parameters.
        """
        self.config = config
        structured_log(logger, logging.INFO, "UncertaintyCalibrator initialized",
                      config_type=type(config).__name__)

    @log_performance
    def calibrate_probabilities(self, 
                              model: Any,
                              X_train: pd.DataFrame,
                              y_train: pd.Series,
                              X_val: pd.DataFrame,
                              method: str = 'sigmoid') -> Tuple[Any, np.ndarray]:
        """
        Calibrate prediction probabilities using sklearn's CalibratedClassifierCV.

        Args:
            model: The trained model to calibrate
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            method: Calibration method ('sigmoid' or 'isotonic')

        Returns:
            Tuple containing:
                - Calibrated classifier
                - Calibrated probabilities for validation set
        """
        structured_log(logger, logging.INFO, "Starting probability calibration",
                      calibration_method=method)
        
        try:
            # Handle various boosting models
            if isinstance(model, (xgb.Booster, lgb.Booster, cb.CatBoost)):
                model = BoostingModelWrapper(model)

            calibrated_classifier = CalibratedClassifierCV(
                base_estimator=model,
                method=method,
                cv='prefit'
            )
            
            calibrated_classifier.fit(X_train, y_train)
            calibrated_probs = calibrated_classifier.predict_proba(X_val)[:, 1]
            
            structured_log(logger, logging.INFO, "Probability calibration completed",
                         output_shape=calibrated_probs.shape)
            
            return calibrated_classifier, calibrated_probs
        
        except Exception as e:
            raise ModelTestingError("Error in probability calibration",
                                  error_message=str(e))

    @log_performance
    def venn_abers_calibration(self,
                              model: Any,
                              X_train: pd.DataFrame,
                              y_train: pd.Series,
                              X_val: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Venn-ABERS probability calibration.

        Args:
            model: The trained model to calibrate
            X_train: Training features
            y_train: Training labels
            X_val: Validation features

        Returns:
            Tuple containing:
                - Lower probabilities
                - Upper probabilities
        """
        structured_log(logger, logging.INFO, "Starting Venn-ABERS calibration")
        
        try:
            # Get uncalibrated predictions
            if isinstance(model, (xgb.Booster, lgb.Booster, cb.CatBoost)):
                if isinstance(model, xgb.Booster):
                    train_probs = model.predict(xgb.DMatrix(X_train))
                    val_probs = model.predict(xgb.DMatrix(X_val))
                elif isinstance(model, lgb.Booster):
                    train_probs = model.predict(X_train)
                    val_probs = model.predict(X_val)
                else:  # CatBoost
                    train_probs = model.predict_proba(X_train)[:, 1]
                    val_probs = model.predict_proba(X_val)[:, 1]

            # Sort training predictions and corresponding labels
            sorted_indices = np.argsort(train_probs)
            sorted_probs = train_probs[sorted_indices]
            sorted_labels = y_train.iloc[sorted_indices]

            # Calculate lower and upper probabilities
            lower_probs = []
            upper_probs = []
            
            for score in val_probs:
                p0, p1 = self._compute_venn_abers_probabilities(
                    score, sorted_probs, sorted_labels)
                lower_probs.append(min(p0, p1))
                upper_probs.append(max(p0, p1))

            structured_log(logger, logging.INFO, "Venn-ABERS calibration completed")
            
            return np.array(lower_probs), np.array(upper_probs)
        
        except Exception as e:
            raise ModelTestingError("Error in Venn-ABERS calibration",
                                  error_message=str(e))

    def _compute_venn_abers_probabilities(self,
                                        score: float,
                                        sorted_scores: np.ndarray,
                                        sorted_labels: np.ndarray) -> Tuple[float, float]:
        """
        Helper method to compute Venn-ABERS probabilities for a single score.
        """
        # Implementation details for Venn-ABERS probability calculation
        # This is a simplified version - you might want to implement the full algorithm
        pass
