"""
Base postprocessor interface for ML output transformations.

Postprocessors apply transformations to model outputs after training,
including calibration, threshold optimization, and other prediction adjustments.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import numpy as np
import pandas as pd
import logging

from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler


class BasePostprocessor(ABC):
    """
    Abstract base class for postprocessing model outputs.

    Postprocessors follow the fit/transform pattern similar to sklearn:
    - fit(): Learn transformation parameters from validation data
    - transform(): Apply learned transformation to new data
    - fit_transform(): Convenience method to fit and transform in one step
    """

    def __init__(self,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        """
        Initialize the base postprocessor.

        Args:
            app_logger: Application logger for structured logging
            error_handler: Error handler for consistent error management
        """
        self.app_logger = app_logger
        self.error_handler = error_handler
        self._is_fitted = False

    @abstractmethod
    def fit(self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            **kwargs) -> 'BasePostprocessor':
        """
        Fit the postprocessor on validation data.

        Args:
            y_pred: Predicted probabilities or values from model
            y_true: True target values
            **kwargs: Additional parameters specific to the postprocessor

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def transform(self,
                  y_pred: np.ndarray,
                  **kwargs) -> np.ndarray:
        """
        Transform predictions using fitted parameters.

        Args:
            y_pred: Predicted probabilities or values to transform
            **kwargs: Additional parameters specific to the postprocessor

        Returns:
            Transformed predictions
        """
        pass

    def fit_transform(self,
                     y_pred: np.ndarray,
                     y_true: np.ndarray,
                     **kwargs) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            y_pred: Predicted probabilities or values
            y_true: True target values
            **kwargs: Additional parameters

        Returns:
            Transformed predictions
        """
        self.fit(y_pred, y_true, **kwargs)
        return self.transform(y_pred, **kwargs)

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get the learned parameters of the postprocessor.

        Returns:
            Dictionary of parameter names to values
        """
        pass

    @abstractmethod
    def set_params(self, params: Dict[str, Any]) -> 'BasePostprocessor':
        """
        Set the parameters of the postprocessor.

        Useful for loading a previously fitted postprocessor.

        Args:
            params: Dictionary of parameter names to values

        Returns:
            Self for method chaining
        """
        pass

    @property
    def is_fitted(self) -> bool:
        """Check if the postprocessor has been fitted."""
        return self._is_fitted

    def validate_fitted(self) -> None:
        """
        Validate that the postprocessor has been fitted.

        Raises:
            RuntimeError: If postprocessor has not been fitted
        """
        if not self._is_fitted:
            raise self.error_handler.create_error_handler(
                'postprocessing',
                "Postprocessor must be fitted before transform"
            )
