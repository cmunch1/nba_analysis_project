"""
Postprocessing module for ML output transformations.

This module provides postprocessing capabilities for model outputs after training,
including probability calibration, threshold optimization, and other prediction adjustments.

Main Components:
    - BasePostprocessor: Abstract base class for all postprocessors
    - ProbabilityCalibrator: Calibrate uncalibrated probability predictions (direct probability method)

Usage:
    from ml_framework.postprocessing import ProbabilityCalibrator

    # Initialize calibrator
    calibrator = ProbabilityCalibrator(
        app_logger=logger,
        error_handler=error_handler,
        method='sigmoid'  # or 'isotonic'
    )

    # Fit and transform on probabilities directly
    calibrated_probs = calibrator.fit_transform(
        y_pred=uncalibrated_probs,
        y_true=y_val
    )
"""

from ml_framework.postprocessing.base_postprocessor import BasePostprocessor
from ml_framework.postprocessing.probability_calibrator import ProbabilityCalibrator

__all__ = [
    'BasePostprocessor',
    'ProbabilityCalibrator'
]
