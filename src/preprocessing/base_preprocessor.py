"""
Base abstract class for preprocessors.

This module defines the base abstract interface for data preprocessing operations.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import pandas as pd

from src.common.data_classes import PreprocessingResults


class BasePreprocessor(ABC):
    """Abstract base class defining the interface for preprocessors."""

    @abstractmethod
    def __init__(self, config, app_logger, app_file_handler, error_handler):
        """
        Initialize preprocessor with required dependencies.
        
        Args:
            config: Configuration manager
            app_logger: Application logger for structured logging
            app_file_handler: File handler for I/O operations
            error_handler: Error handler for standardized error management
        """
        pass

    @property
    @abstractmethod
    def log_performance(self):
        """Get the performance logging decorator."""
        pass

    @abstractmethod
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                    model_name: str = None, preprocessing_results: PreprocessingResults = None) -> Tuple[pd.DataFrame, PreprocessingResults]:
        """
        Fit preprocessor to data and transform it.
        
        Args:
            X: Input features DataFrame
            y: Optional target Series for supervised preprocessing steps
            model_name: Name of the model to get specific preprocessing config
            preprocessing_results: Optional existing preprocessing results to update
            
        Returns:
            Tuple containing:
            - Transformed DataFrame
            - Updated preprocessing results
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the fitted preprocessor.
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Transformed DataFrame
        """
        pass