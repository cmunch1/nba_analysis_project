"""
Postprocessing module for ML output transformations.

This module provides postprocessing capabilities for model outputs after training,
including probability calibration, threshold optimization, and other prediction adjustments.

Main Components:
    - BasePostprocessor: Abstract base class for all postprocessors
    - ProbabilityCalibrator: Calibrate uncalibrated probability predictions (direct probability method)
    - CalibrationOptimizer: Automatically search for best calibration method and parameters
    - ConformalPredictor: Build conformal prediction sets and probability intervals on calibrated probabilities

Usage:
    from ml_framework.postprocessing import ProbabilityCalibrator, CalibrationOptimizer

    # Manual calibration
    calibrator = ProbabilityCalibrator(
        app_logger=logger,
        error_handler=error_handler,
        method='sigmoid'  # or 'isotonic'
    )
    calibrated_probs = calibrator.fit_transform(
        y_pred=uncalibrated_probs,
        y_true=y_val
    )

    # Automatic optimization
    optimizer = CalibrationOptimizer(
        app_logger=logger,
        error_handler=error_handler,
        methods=['sigmoid', 'isotonic'],
        evaluation_bins=[5, 10, 15, 20],
        selection_metric='brier_score'
    )
    best_calibrator, results = optimizer.optimize(
        y_pred=uncalibrated_probs,
        y_true=y_val
    )
"""

from ml_framework.postprocessing.base_postprocessor import BasePostprocessor
from ml_framework.postprocessing.probability_calibrator import ProbabilityCalibrator
from ml_framework.postprocessing.calibration_optimizer import CalibrationOptimizer
from ml_framework.postprocessing.conformal_predictor import ConformalPredictor

__all__ = [
    'BasePostprocessor',
    'ProbabilityCalibrator',
    'CalibrationOptimizer',
    'ConformalPredictor'
]
