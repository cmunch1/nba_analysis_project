"""
Conformal prediction postprocessor for calibrated probabilities.

Provides prediction sets and probability intervals on top of calibrated
binary classification probabilities using split conformal prediction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Literal

import numpy as np

from ml_framework.postprocessing.base_postprocessor import BasePostprocessor
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler


ConformalMethod = Literal['split']
ScoreFunction = Literal['probability_shortfall', 'absolute_error']


@dataclass
class ConformalQuantiles:
    """Container for cached conformal quantiles."""
    prediction_set: float
    probability_interval: Optional[float]


class ConformalPredictor(BasePostprocessor):
    """
    Split conformal predictor for calibrated binary probabilities.

    Given calibrated probabilities and true labels, computes nonconformity
    scores and stores quantiles needed to build prediction sets and
    probability intervals for new predictions.
    """

    def __init__(self,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler,
                 method: ConformalMethod = 'split',
                 score_function: ScoreFunction = 'probability_shortfall',
                 alpha_prediction_set: float = 0.1,
                 alpha_probability_interval: float = 0.2,
                 class_labels: Optional[List[str]] = None,
                 allow_empty_set: bool = False):
        """
        Initialize conformal predictor.

        Args:
            app_logger: Logger for structured logging
            error_handler: Shared error handler
            method: Conformal method ('split' currently supported)
            score_function: Nonconformity score used for prediction sets
            alpha_prediction_set: Target miscoverage for prediction sets
            alpha_probability_interval: Target miscoverage for probability intervals
            class_labels: Optional ordered labels [negative, positive]
            allow_empty_set: Whether to allow empty prediction sets
        """
        super().__init__(app_logger, error_handler)
        self.method = method
        self.score_function = score_function
        self.alpha_prediction_set = alpha_prediction_set
        self.alpha_probability_interval = alpha_probability_interval
        self.class_labels = list(class_labels) if class_labels is not None else ['negative', 'positive']
        self.allow_empty_set = allow_empty_set

        self.quantiles_: Optional[ConformalQuantiles] = None
        self.metrics_: Dict[str, Any] = {}
        self.n_calibration_samples_: int = 0

        self.app_logger.structured_log(
            logging.INFO,
            "ConformalPredictor initialized",
            method=method,
            score_function=score_function,
            alpha_prediction_set=alpha_prediction_set,
            alpha_probability_interval=alpha_probability_interval
        )

    def fit(self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            **kwargs) -> 'ConformalPredictor':
        """
        Fit conformal predictor using calibrated probabilities and labels.

        Args:
            y_pred: Calibrated probabilities for positive class (shape: [n_samples])
            y_true: Binary true labels (0/1)

        Returns:
            Self for method chaining
        """
        self.app_logger.structured_log(
            logging.INFO,
            "Fitting ConformalPredictor",
            method=self.method,
            score_function=self.score_function,
            n_samples=len(y_true)
        )

        try:
            if self.method != 'split':
                raise ValueError(f"Unsupported conformal method: {self.method}")

            if len(y_pred) != len(y_true):
                raise ValueError(
                    f"Length mismatch: y_pred has {len(y_pred)} samples, "
                    f"y_true has {len(y_true)} samples"
                )

            y_pred = np.clip(y_pred.astype(float), 1e-12, 1 - 1e-12)
            y_true = y_true.astype(int)
            self.n_calibration_samples_ = len(y_true)

            # Compute nonconformity scores
            prediction_set_scores = self._compute_scores(y_pred, y_true, for_sets=True)
            interval_scores = np.abs(y_true - y_pred)

            # Cache quantiles
            q_set = self._conformal_quantile(prediction_set_scores, self.alpha_prediction_set)
            q_interval = None
            if self.alpha_probability_interval is not None:
                q_interval = self._conformal_quantile(interval_scores, self.alpha_probability_interval)

            self.quantiles_ = ConformalQuantiles(
                prediction_set=float(q_set),
                probability_interval=float(q_interval) if q_interval is not None else None
            )

            # Store diagnostics
            metrics = self._compute_calibration_metrics(
                y_pred=y_pred,
                y_true=y_true,
                prediction_set_scores=prediction_set_scores,
                interval_scores=interval_scores,
                q_set=q_set,
                q_interval=q_interval
            )
            self.metrics_ = metrics

            self._is_fitted = True

            self.app_logger.structured_log(
                logging.INFO,
                "ConformalPredictor fitted successfully",
                q_prediction_set=self.quantiles_.prediction_set,
                q_probability_interval=self.quantiles_.probability_interval
            )

            return self

        except Exception as exc:
            raise self.error_handler.create_error_handler(
                'postprocessing',
                "Error fitting ConformalPredictor",
                original_error=str(exc)
            )

    def transform(self,
                  y_pred: np.ndarray,
                  **kwargs) -> Dict[str, Any]:
        """
        Generate conformal outputs for new probabilities.

        Args:
            y_pred: Calibrated probabilities for positive class

        Returns:
            Dictionary with prediction sets and probability intervals
        """
        self.validate_fitted()

        try:
            y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-12, 1 - 1e-12)
            prediction_sets = self._build_prediction_sets(y_pred)
            probability_intervals = self._build_probability_intervals(y_pred)

            return {
                'prediction_sets': prediction_sets,
                'probability_intervals': probability_intervals
            }

        except Exception as exc:
            raise self.error_handler.create_error_handler(
                'postprocessing',
                "Error transforming with ConformalPredictor",
                original_error=str(exc)
            )

    def predict_set(self, y_pred: np.ndarray) -> List[List[str]]:
        """Convenience wrapper returning only prediction sets."""
        self.validate_fitted()
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-12, 1 - 1e-12)
        return self._build_prediction_sets(y_pred)

    def predict_interval(self, y_pred: np.ndarray) -> Optional[np.ndarray]:
        """Convenience wrapper returning only probability intervals."""
        self.validate_fitted()
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-12, 1 - 1e-12)
        return self._build_probability_intervals(y_pred)

    def get_params(self) -> Dict[str, Any]:
        """Return fitted parameters for persistence."""
        if not self._is_fitted or self.quantiles_ is None:
            return {}

        return {
            'method': self.method,
            'score_function': self.score_function,
            'alpha_prediction_set': self.alpha_prediction_set,
            'alpha_probability_interval': self.alpha_probability_interval,
            'class_labels': self.class_labels,
            'allow_empty_set': self.allow_empty_set,
            'quantiles': {
                'prediction_set': self.quantiles_.prediction_set,
                'probability_interval': self.quantiles_.probability_interval
            },
            'metrics': self.metrics_,
            'n_calibration_samples': self.n_calibration_samples_
        }

    def set_params(self, params: Dict[str, Any]) -> 'ConformalPredictor':
        """Restore fitted parameters from persisted state."""
        self.method = params.get('method', self.method)
        self.score_function = params.get('score_function', self.score_function)
        self.alpha_prediction_set = params.get('alpha_prediction_set', self.alpha_prediction_set)
        self.alpha_probability_interval = params.get('alpha_probability_interval', self.alpha_probability_interval)
        self.class_labels = params.get('class_labels', self.class_labels)
        self.allow_empty_set = params.get('allow_empty_set', self.allow_empty_set)

        quantiles = params.get('quantiles')
        if quantiles is not None:
            self.quantiles_ = ConformalQuantiles(
                prediction_set=float(quantiles['prediction_set']),
                probability_interval=float(quantiles['probability_interval']) if quantiles.get('probability_interval') is not None else None
            )
            self._is_fitted = True

        self.metrics_ = params.get('metrics', {})
        self.n_calibration_samples_ = params.get('n_calibration_samples', 0)

        return self

    # Internal helpers -----------------------------------------------------

    def _compute_scores(self,
                        y_pred: np.ndarray,
                        y_true: np.ndarray,
                        for_sets: bool) -> np.ndarray:
        """Compute nonconformity scores."""
        if self.score_function == 'probability_shortfall':
            prob_true = np.where(y_true == 1, y_pred, 1 - y_pred)
            scores = 1.0 - prob_true
        elif self.score_function == 'absolute_error':
            scores = np.abs(y_true - y_pred)
        else:
            raise ValueError(f"Unknown score_function: {self.score_function}")

        if for_sets and np.any(scores < 0):
            scores = np.maximum(scores, 0.0)

        return scores

    @staticmethod
    def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
        """Calculate conformal quantile using the standard order statistic."""
        if scores.ndim != 1:
            raise ValueError("scores must be 1-dimensional")
        n = len(scores)
        if n == 0:
            raise ValueError("Cannot compute conformal quantile with no calibration data")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")

        k = int(np.ceil((1 - alpha) * (n + 1)))
        k = min(max(k, 1), n)
        return float(np.partition(scores, k - 1)[k - 1])

    def _build_prediction_sets(self, y_pred: np.ndarray) -> List[List[str]]:
        """Construct conformal prediction sets for each instance."""
        labels = self.class_labels
        if len(labels) != 2:
            raise ValueError("class_labels must contain exactly two entries for binary classification")

        q_set = self.quantiles_.prediction_set if self.quantiles_ else None
        if q_set is None:
            raise RuntimeError("Prediction set quantile not fitted")

        prediction_sets: List[List[str]] = []
        for p in y_pred:
            scores = {
                labels[1]: 1.0 - p,  # positive class
                labels[0]: 1.0 - (1.0 - p)
            }

            included = [label for label, score in scores.items() if score <= q_set]

            if not included and not self.allow_empty_set:
                included = labels.copy()

            prediction_sets.append(included)

        return prediction_sets

    def _build_probability_intervals(self, y_pred: np.ndarray) -> Optional[np.ndarray]:
        """Construct symmetric probability intervals around calibrated probability."""
        if self.quantiles_ is None or self.quantiles_.probability_interval is None:
            return None

        q_interval = self.quantiles_.probability_interval
        lower = np.clip(y_pred - q_interval, 0.0, 1.0)
        upper = np.clip(y_pred + q_interval, 0.0, 1.0)
        intervals = np.stack([lower, upper], axis=1)
        return intervals

    def _compute_calibration_metrics(self,
                                     y_pred: np.ndarray,
                                     y_true: np.ndarray,
                                     prediction_set_scores: np.ndarray,
                                     interval_scores: np.ndarray,
                                     q_set: float,
                                     q_interval: Optional[float]) -> Dict[str, Any]:
        """Calculate empirical coverage diagnostics on calibration data."""
        prediction_sets_cal = self._build_prediction_sets(y_pred)
        true_labels = np.where(y_true == 1, self.class_labels[1], self.class_labels[0])
        coverage = np.mean([
            1.0 if true_label in pred_set else 0.0
            for true_label, pred_set in zip(true_labels, prediction_sets_cal)
        ])

        metrics: Dict[str, Any] = {
            'alpha_prediction_set': self.alpha_prediction_set,
            'target_coverage_prediction_set': 1 - self.alpha_prediction_set,
            'empirical_coverage_prediction_set': float(coverage),
            'quantile_prediction_set': float(q_set),
            'mean_score_prediction_set': float(np.mean(prediction_set_scores)),
            'n_calibration_samples': int(len(y_true))
        }

        if q_interval is not None:
            metrics.update({
                'alpha_probability_interval': self.alpha_probability_interval,
                'target_coverage_probability_interval': 1 - self.alpha_probability_interval,
                'quantile_probability_interval': float(q_interval),
                'mean_score_probability_interval': float(np.mean(interval_scores)),
                'empirical_radius_probability_interval': float(q_interval),
                'empirical_interval_width': float(2 * q_interval)
            })

        return metrics
