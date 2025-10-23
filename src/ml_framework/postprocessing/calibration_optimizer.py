"""
Calibration hyperparameter optimizer for probability calibration.

Searches over calibration methods and parameters to find the optimal
calibration configuration based on validation metrics.
"""

import logging
from typing import Dict, Any, List, Optional, Literal, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import brier_score_loss, log_loss

from ml_framework.postprocessing.probability_calibrator import ProbabilityCalibrator
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler


class CalibrationOptimizer:
    """
    Optimize calibration method and parameters via grid search.

    This class searches over different calibration methods (sigmoid, isotonic)
    and evaluation configurations to find the best calibration approach based
    on metrics like Brier score, ECE, or log loss.

    Production ML systems typically use this approach to automate calibration
    selection during model training/validation rather than manually choosing
    calibration parameters.

    Usage:
        optimizer = CalibrationOptimizer(
            app_logger=logger,
            error_handler=error_handler,
            methods=['sigmoid', 'isotonic'],
            evaluation_bins=[5, 10, 15, 20],
            selection_metric='brier_score',
            cv_folds=3
        )

        # Find best calibrator
        best_calibrator, results = optimizer.optimize(
            y_pred=uncalibrated_probs,
            y_true=y_val
        )

        # Use best calibrator
        calibrated_probs = best_calibrator.transform(test_probs)
    """

    def __init__(self,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler,
                 methods: List[Literal['sigmoid', 'isotonic']] = None,
                 evaluation_bins: List[int] = None,
                 selection_metric: Literal['brier_score', 'ece', 'log_loss'] = 'brier_score',
                 cv_folds: int = 3,
                 random_state: int = 42):
        """
        Initialize the calibration optimizer.

        Args:
            app_logger: Application logger for structured logging
            error_handler: Error handler for consistent error management
            methods: List of calibration methods to try (default: ['sigmoid', 'isotonic'])
            evaluation_bins: List of bin sizes for ECE evaluation (default: [10])
            selection_metric: Metric to use for selection ('brier_score', 'ece', 'log_loss')
            cv_folds: Number of cross-validation folds for robust evaluation
            random_state: Random seed for reproducibility
        """
        self.app_logger = app_logger
        self.error_handler = error_handler
        self.methods = methods or ['sigmoid', 'isotonic']
        self.evaluation_bins = evaluation_bins or [10]
        self.selection_metric = selection_metric
        self.cv_folds = cv_folds
        self.random_state = random_state

        # Results storage
        self.best_calibrator_: Optional[ProbabilityCalibrator] = None
        self.best_config_: Optional[Dict[str, Any]] = None
        self.optimization_results_: List[Dict[str, Any]] = []

        self.app_logger.structured_log(
            logging.INFO,
            "CalibrationOptimizer initialized",
            methods=self.methods,
            evaluation_bins=self.evaluation_bins,
            selection_metric=self.selection_metric,
            cv_folds=self.cv_folds
        )

    def optimize(self,
                 y_pred: np.ndarray,
                 y_true: np.ndarray,
                 use_cv: bool = True) -> Tuple[ProbabilityCalibrator, List[Dict[str, Any]]]:
        """
        Optimize calibration method and parameters.

        Args:
            y_pred: Uncalibrated predicted probabilities (from validation set)
            y_true: True target values (binary: 0 or 1)
            use_cv: Whether to use cross-validation (recommended for robust selection)

        Returns:
            Tuple of (best_calibrator, optimization_results)
            - best_calibrator: Fitted ProbabilityCalibrator with best configuration
            - optimization_results: List of dicts with results for all configurations
        """
        self.app_logger.structured_log(
            logging.INFO,
            "Starting calibration optimization",
            n_samples=len(y_true),
            n_methods=len(self.methods),
            n_bins_configs=len(self.evaluation_bins),
            use_cv=use_cv
        )

        try:
            # Validate inputs
            if len(y_pred) != len(y_true):
                raise ValueError(
                    f"Length mismatch: y_pred has {len(y_pred)} samples, "
                    f"y_true has {len(y_true)} samples"
                )

            # Convert to numpy arrays to avoid pandas indexing issues during CV
            # KFold.split() returns integer position indices, but pandas Series/DataFrames
            # may have non-integer or non-sequential indices, causing indexing errors
            if isinstance(y_pred, (pd.Series, pd.DataFrame)):
                self.app_logger.structured_log(
                    logging.DEBUG,
                    "Converting y_pred from pandas to numpy for CV compatibility",
                    original_type=type(y_pred).__name__
                )
                y_pred = y_pred.values if isinstance(y_pred, pd.Series) else y_pred.values.flatten()

            if isinstance(y_true, (pd.Series, pd.DataFrame)):
                self.app_logger.structured_log(
                    logging.DEBUG,
                    "Converting y_true from pandas to numpy for CV compatibility",
                    original_type=type(y_true).__name__
                )
                y_true = y_true.values if isinstance(y_true, pd.Series) else y_true.values.flatten()

            # Ensure we have numpy arrays
            y_pred = np.asarray(y_pred)
            y_true = np.asarray(y_true)

            # Ensure probabilities are in valid range
            y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)

            # Build search grid
            search_grid = self._build_search_grid()

            self.app_logger.structured_log(
                logging.INFO,
                "Built search grid",
                n_configurations=len(search_grid)
            )

            # Evaluate each configuration
            results = []
            for config in search_grid:
                if use_cv:
                    config_results = self._evaluate_config_cv(config, y_pred, y_true)
                else:
                    config_results = self._evaluate_config_simple(config, y_pred, y_true)

                results.append(config_results)

                self.app_logger.structured_log(
                    logging.DEBUG,
                    "Evaluated configuration",
                    method=config['method'],
                    eval_bins=config['eval_bins'],
                    mean_score=config_results['mean_score'],
                    std_score=config_results.get('std_score', 0.0)
                )

            # Store all results
            self.optimization_results_ = results

            # Select best configuration
            best_result = self._select_best_config(results)
            self.best_config_ = best_result['config']

            self.app_logger.structured_log(
                logging.INFO,
                "Best calibration configuration selected",
                method=self.best_config_['method'],
                eval_bins=self.best_config_['eval_bins'],
                selection_metric=self.selection_metric,
                best_score=best_result['mean_score'],
                std_score=best_result.get('std_score', 0.0)
            )

            # Fit best calibrator on full calibration set
            self.best_calibrator_ = ProbabilityCalibrator(
                app_logger=self.app_logger,
                error_handler=self.error_handler,
                method=self.best_config_['method']
            )
            self.best_calibrator_.fit(y_pred=y_pred, y_true=y_true)

            self.app_logger.structured_log(
                logging.INFO,
                "Calibration optimization complete",
                best_method=self.best_config_['method'],
                n_configurations_evaluated=len(results)
            )

            return self.best_calibrator_, self.optimization_results_

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'postprocessing',
                "Error during calibration optimization",
                original_error=str(e),
                n_samples=len(y_true)
            )

    def _build_search_grid(self) -> List[Dict[str, Any]]:
        """
        Build the search grid of configurations to evaluate.

        Returns:
            List of configuration dictionaries
        """
        grid = []
        for method in self.methods:
            for eval_bins in self.evaluation_bins:
                grid.append({
                    'method': method,
                    'eval_bins': eval_bins
                })
        return grid

    def _evaluate_config_cv(self,
                           config: Dict[str, Any],
                           y_pred: np.ndarray,
                           y_true: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a configuration using cross-validation.

        This provides robust estimates by averaging performance across
        multiple calibration train/test splits.

        Args:
            config: Configuration dictionary with 'method' and 'eval_bins'
            y_pred: Uncalibrated probabilities
            y_true: True labels

        Returns:
            Dictionary with mean and std of evaluation metric
        """
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = []

        for train_idx, test_idx in kf.split(y_pred):
            # Split calibration data
            y_pred_train, y_pred_test = y_pred[train_idx], y_pred[test_idx]
            y_true_train, y_true_test = y_true[train_idx], y_true[test_idx]

            # Fit calibrator on train fold
            calibrator = ProbabilityCalibrator(
                app_logger=self.app_logger,
                error_handler=self.error_handler,
                method=config['method']
            )
            calibrator.fit(y_pred=y_pred_train, y_true=y_true_train)

            # Evaluate on test fold
            y_pred_calibrated = calibrator.transform(y_pred=y_pred_test)
            score = self._calculate_metric(
                y_true_test,
                y_pred_calibrated,
                config['eval_bins']
            )
            scores.append(score)

        return {
            'config': config,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'fold_scores': scores,
            'method': config['method'],
            'eval_bins': config['eval_bins']
        }

    def _evaluate_config_simple(self,
                               config: Dict[str, Any],
                               y_pred: np.ndarray,
                               y_true: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a configuration using simple train/test split.

        Faster but less robust than CV. Use when calibration set is large.

        Args:
            config: Configuration dictionary with 'method' and 'eval_bins'
            y_pred: Uncalibrated probabilities
            y_true: True labels

        Returns:
            Dictionary with evaluation metric score
        """
        # Use 70/30 split for calibration train/test
        n_samples = len(y_pred)
        n_train = int(0.7 * n_samples)

        # Shuffle indices
        rng = np.random.RandomState(self.random_state)
        indices = rng.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        # Split data
        y_pred_train, y_pred_test = y_pred[train_idx], y_pred[test_idx]
        y_true_train, y_true_test = y_true[train_idx], y_true[test_idx]

        # Fit calibrator
        calibrator = ProbabilityCalibrator(
            app_logger=self.app_logger,
            error_handler=self.error_handler,
            method=config['method']
        )
        calibrator.fit(y_pred=y_pred_train, y_true=y_true_train)

        # Evaluate
        y_pred_calibrated = calibrator.transform(y_pred=y_pred_test)
        score = self._calculate_metric(
            y_true_test,
            y_pred_calibrated,
            config['eval_bins']
        )

        return {
            'config': config,
            'mean_score': score,
            'std_score': 0.0,
            'method': config['method'],
            'eval_bins': config['eval_bins']
        }

    def _calculate_metric(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         n_bins: int) -> float:
        """
        Calculate the selection metric.

        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            n_bins: Number of bins for ECE calculation

        Returns:
            Metric score (lower is better for all metrics)
        """
        if self.selection_metric == 'brier_score':
            return brier_score_loss(y_true, y_pred)
        elif self.selection_metric == 'log_loss':
            return log_loss(y_true, y_pred)
        elif self.selection_metric == 'ece':
            return self._calculate_ece(y_true, y_pred, n_bins)
        else:
            raise ValueError(f"Unknown selection metric: {self.selection_metric}")

    def _calculate_ece(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      n_bins: int) -> float:
        """
        Calculate Expected Calibration Error (ECE).

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
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)

            bin_size = np.sum(in_bin)
            if bin_size > 0:
                bin_accuracy = np.mean(y_true[in_bin])
                bin_confidence = np.mean(y_pred[in_bin])
                ece += (bin_size / len(y_true)) * np.abs(bin_accuracy - bin_confidence)

        return ece

    def _select_best_config(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the best configuration from results.

        Uses mean score (lower is better for all metrics).
        In case of ties, prefers sigmoid over isotonic for stability.

        Args:
            results: List of result dictionaries

        Returns:
            Best result dictionary
        """
        # Sort by mean score (ascending - lower is better)
        sorted_results = sorted(results, key=lambda x: x['mean_score'])

        # Get best score
        best_score = sorted_results[0]['mean_score']

        # Find all configs within tolerance (1% of best score)
        tolerance = 0.01 * abs(best_score) if best_score != 0 else 0.001
        candidates = [r for r in sorted_results if abs(r['mean_score'] - best_score) <= tolerance]

        # If multiple candidates, prefer sigmoid for stability
        if len(candidates) > 1:
            sigmoid_candidates = [c for c in candidates if c['method'] == 'sigmoid']
            if sigmoid_candidates:
                return sigmoid_candidates[0]

        return sorted_results[0]

    def get_optimization_summary(self) -> pd.DataFrame:
        """
        Get a summary of all evaluated configurations.

        Returns:
            DataFrame with all configurations and their scores
        """
        if not self.optimization_results_:
            return pd.DataFrame()

        summary_data = []
        for result in self.optimization_results_:
            summary_data.append({
                'method': result['method'],
                'eval_bins': result['eval_bins'],
                'mean_score': result['mean_score'],
                'std_score': result.get('std_score', 0.0),
                'metric': self.selection_metric,
                'is_best': (result['config'] == self.best_config_)
            })

        df = pd.DataFrame(summary_data)
        df = df.sort_values('mean_score', ascending=True)
        return df

    def get_best_calibrator(self) -> Optional[ProbabilityCalibrator]:
        """
        Get the best calibrator found during optimization.

        Returns:
            Fitted ProbabilityCalibrator or None if optimize() hasn't been called
        """
        return self.best_calibrator_

    def get_best_config(self) -> Optional[Dict[str, Any]]:
        """
        Get the best configuration found during optimization.

        Returns:
            Configuration dictionary or None if optimize() hasn't been called
        """
        return self.best_config_
