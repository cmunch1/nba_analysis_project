"""
Probability calibration postprocessor for ML models.

Provides methods to calibrate probability predictions from models that
produce uncalibrated probabilities (e.g., XGBoost, LightGBM, Random Forest).

Uses direct probability calibration approach (not model wrapping) for maximum
compatibility with all sklearn versions and model types.
"""

import logging
from typing import Dict, Any, Optional, Literal, Union
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from ml_framework.postprocessing.base_postprocessor import BasePostprocessor
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler


class ProbabilityCalibrator(BasePostprocessor):
    """
    Calibrate probability predictions using direct probability transformation.

    Supports:
    - Platt scaling (sigmoid): Logistic regression on predicted probabilities
    - Isotonic regression: Non-parametric monotonic calibration

    This implementation uses direct probability calibration (not model wrapping)
    for maximum compatibility with all sklearn versions and model types.

    Usage:
        calibrator = ProbabilityCalibrator(app_logger, error_handler, method='sigmoid')

        # Fit on validation set probabilities
        calibrator.fit(
            y_pred=uncalibrated_probs,
            y_true=y_val
        )

        # Transform new predictions
        calibrated_probs = calibrator.transform(test_probs)

        # Or in one step
        calibrated_probs = calibrator.fit_transform(
            y_pred=uncalibrated_probs,
            y_true=y_val
        )
    """

    def __init__(self,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler,
                 method: Literal['sigmoid', 'isotonic'] = 'sigmoid'):
        """
        Initialize the probability calibrator.

        Args:
            app_logger: Application logger for structured logging
            error_handler: Error handler for consistent error management
            method: Calibration method ('sigmoid' for Platt scaling, 'isotonic')
        """
        super().__init__(app_logger, error_handler)
        self.method = method
        self.calibrator_: Optional[Union[LogisticRegression, IsotonicRegression]] = None
        self.calibration_metrics_ = {}

        self.app_logger.structured_log(
            logging.INFO,
            "ProbabilityCalibrator initialized (direct probability method)",
            method=method
        )

    def fit(self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            model: Optional[Any] = None,
            X_cal: Optional[pd.DataFrame] = None,
            **kwargs) -> 'ProbabilityCalibrator':
        """
        Fit the calibration model on predicted probabilities.

        Args:
            y_pred: Uncalibrated predicted probabilities
            y_true: True target values (binary: 0 or 1)
            model: Optional trained model (ignored in direct probability mode)
            X_cal: Optional calibration features (ignored in direct probability mode)
            **kwargs: Additional parameters

        Returns:
            Self for method chaining
        """
        self.app_logger.structured_log(
            logging.INFO,
            "Fitting probability calibrator (direct probability method)",
            method=self.method,
            n_samples=len(y_true)
        )

        try:
            # Validate inputs
            if len(y_pred) != len(y_true):
                raise ValueError(
                    f"Length mismatch: y_pred has {len(y_pred)} samples, "
                    f"y_true has {len(y_true)} samples"
                )

            # Ensure probabilities are in valid range [0, 1]
            y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)

            # Fit calibrator based on method
            if self.method == 'sigmoid':
                # Platt scaling: logistic regression on predicted probabilities
                self.calibrator_ = LogisticRegression(solver='lbfgs', max_iter=1000)
                # Reshape to 2D array for sklearn
                self.calibrator_.fit(y_pred.reshape(-1, 1), y_true)

                # Get calibrated probabilities
                calibrated_probs = self.calibrator_.predict_proba(y_pred.reshape(-1, 1))[:, 1]

            elif self.method == 'isotonic':
                # Isotonic regression: non-parametric monotonic calibration
                self.calibrator_ = IsotonicRegression(out_of_bounds='clip')
                self.calibrator_.fit(y_pred, y_true)

                # Get calibrated probabilities
                calibrated_probs = self.calibrator_.transform(y_pred)

            else:
                raise ValueError(
                    f"Unknown calibration method: {self.method}. "
                    f"Must be 'sigmoid' or 'isotonic'"
                )

            # Calculate and store calibration metrics
            self._calculate_calibration_metrics(y_pred, calibrated_probs, y_true)

            self._is_fitted = True

            self.app_logger.structured_log(
                logging.INFO,
                "Probability calibrator fitted successfully",
                method=self.method,
                brier_score_before=self.calibration_metrics_.get('brier_score_before'),
                brier_score_after=self.calibration_metrics_.get('brier_score_after'),
                improvement=self.calibration_metrics_.get('brier_score_improvement')
            )

            return self

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'postprocessing',
                "Error fitting probability calibrator",
                original_error=str(e),
                method=self.method,
                n_samples=len(y_true)
            )

    def transform(self,
                  y_pred: Optional[np.ndarray] = None,
                  X: Optional[pd.DataFrame] = None,
                  **kwargs) -> np.ndarray:
        """
        Transform uncalibrated probabilities using fitted calibrator.

        Args:
            y_pred: Uncalibrated probabilities to calibrate
            X: Features (ignored in direct probability mode, kept for API compatibility)
            **kwargs: Additional parameters

        Returns:
            Calibrated probabilities
        """
        self.validate_fitted()

        try:
            if y_pred is None:
                raise ValueError("y_pred must be provided for probability calibration")

            # Ensure probabilities are in valid range [0, 1]
            y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)

            # Transform based on calibration method
            if self.method == 'sigmoid':
                # Use fitted logistic regression
                calibrated_probs = self.calibrator_.predict_proba(y_pred.reshape(-1, 1))[:, 1]
            elif self.method == 'isotonic':
                # Use fitted isotonic regression
                calibrated_probs = self.calibrator_.transform(y_pred)
            else:
                raise ValueError(f"Unknown calibration method: {self.method}")

            self.app_logger.structured_log(
                logging.DEBUG,
                "Probabilities calibrated",
                method=self.method,
                n_samples=len(calibrated_probs)
            )

            return calibrated_probs

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'postprocessing',
                "Error transforming probabilities",
                original_error=str(e),
                method=self.method
            )

    def get_params(self) -> Dict[str, Any]:
        """
        Get the learned parameters of the calibrator.

        Returns:
            Dictionary containing calibration parameters and metrics
        """
        if not self._is_fitted:
            return {}

        return {
            'method': self.method,
            'calibration_metrics': self.calibration_metrics_,
            'is_fitted': self._is_fitted
        }

    def set_params(self, params: Dict[str, Any]) -> 'ProbabilityCalibrator':
        """
        Set the parameters of the calibrator.

        Args:
            params: Dictionary of parameter names to values

        Returns:
            Self for method chaining
        """
        self.method = params.get('method', self.method)
        self.calibration_metrics_ = params.get('calibration_metrics', {})
        self._is_fitted = params.get('is_fitted', False)

        return self

    def get_calibration_curve(self,
                              y_true: np.ndarray,
                              y_pred_uncalibrated: np.ndarray,
                              y_pred_calibrated: np.ndarray,
                              n_bins: int = 10) -> Dict[str, np.ndarray]:
        """
        Calculate calibration curve data for visualization.

        Args:
            y_true: True labels
            y_pred_uncalibrated: Uncalibrated probabilities
            y_pred_calibrated: Calibrated probabilities
            n_bins: Number of bins for calibration curve

        Returns:
            Dictionary with calibration curve data
        """
        from sklearn.calibration import calibration_curve

        # Calculate curves
        prob_true_uncal, prob_pred_uncal = calibration_curve(
            y_true, y_pred_uncalibrated, n_bins=n_bins, strategy='uniform'
        )
        prob_true_cal, prob_pred_cal = calibration_curve(
            y_true, y_pred_calibrated, n_bins=n_bins, strategy='uniform'
        )

        return {
            'uncalibrated': {
                'prob_true': prob_true_uncal,
                'prob_pred': prob_pred_uncal
            },
            'calibrated': {
                'prob_true': prob_true_cal,
                'prob_pred': prob_pred_cal
            },
            'n_bins': n_bins
        }

    def _calculate_calibration_metrics(self,
                                      y_pred_before: np.ndarray,
                                      y_pred_after: np.ndarray,
                                      y_true: np.ndarray) -> None:
        """Calculate and store calibration quality metrics."""
        # Brier score (lower is better)
        brier_before = brier_score_loss(y_true, y_pred_before)
        brier_after = brier_score_loss(y_true, y_pred_after)

        # Log loss (lower is better)
        logloss_before = log_loss(y_true, y_pred_before)
        logloss_after = log_loss(y_true, y_pred_after)

        # Expected Calibration Error (ECE)
        ece_before = self._calculate_ece(y_true, y_pred_before)
        ece_after = self._calculate_ece(y_true, y_pred_after)

        self.calibration_metrics_ = {
            'brier_score_before': float(brier_before),
            'brier_score_after': float(brier_after),
            'brier_score_improvement': float(brier_before - brier_after),
            'log_loss_before': float(logloss_before),
            'log_loss_after': float(logloss_after),
            'log_loss_improvement': float(logloss_before - logloss_after),
            'ece_before': float(ece_before),
            'ece_after': float(ece_after),
            'ece_improvement': float(ece_before - ece_after)
        }

    def _calculate_ece(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        ECE measures the difference between predicted probabilities
        and actual outcomes across different probability bins.

        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            n_bins: Number of bins

        Returns:
            ECE score (lower is better)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            # Find predictions in this bin
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)

            # Calculate bin metrics
            bin_size = np.sum(in_bin)
            if bin_size > 0:
                bin_accuracy = np.mean(y_true[in_bin])
                bin_confidence = np.mean(y_pred[in_bin])
                ece += (bin_size / len(y_true)) * np.abs(bin_accuracy - bin_confidence)

        return ece
