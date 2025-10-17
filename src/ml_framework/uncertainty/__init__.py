"""
Uncertainty quantification module.

DEPRECATED: Calibration functionality has been moved to ml_framework.postprocessing.
This module now focuses on true uncertainty quantification methods.

For probability calibration, use:
    from ml_framework.postprocessing import ProbabilityCalibrator

This module is reserved for:
    - Conformal prediction
    - Uncertainty estimation
    - Prediction intervals
    - Bayesian uncertainty quantification

Note: uncertainty_calibrator.py is deprecated and will be removed in a future version.
"""

# Deprecated - use ml_framework.postprocessing.ProbabilityCalibrator instead
from ml_framework.uncertainty.uncertainty_calibrator import (
    UncertaintyCalibrator,
    BoostingModelWrapper
)

import warnings

warnings.warn(
    "UncertaintyCalibrator is deprecated. "
    "Use ml_framework.postprocessing.ProbabilityCalibrator instead. "
    "This module will be refactored to focus on uncertainty quantification.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'UncertaintyCalibrator',  # Deprecated
    'BoostingModelWrapper'     # Deprecated
]