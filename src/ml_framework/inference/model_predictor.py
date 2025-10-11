"""
Model predictor for inference using registered models.

Handles loading trained models with their preprocessing artifacts and
applying the complete inference pipeline.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from pathlib import Path

from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler
from ml_framework.model_registry.base_model_registry import BaseModelRegistry
from ml_framework.preprocessing.preprocessor import Preprocessor


class ModelPredictor:
    """
    Model predictor for inference using registered models.

    Loads trained models with their preprocessing artifacts from a model registry
    and provides a clean interface for making predictions on new data.

    Design:
    - Uses dependency injection for model registry (MLflow, custom, etc.)
    - Automatically applies preprocessing if preprocessor artifact exists
    - Supports both probability and class predictions
    - Handles feature schema validation
    """

    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler,
                 model_registry: BaseModelRegistry):
        """
        Initialize model predictor.

        Args:
            config: Configuration manager
            app_logger: Application logger for structured logging
            error_handler: Error handler for standardized error management
            model_registry: Model registry implementation (MLflow, custom, etc.)
        """
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler
        self.model_registry = model_registry

        # Loaded model components
        self.model = None
        self.preprocessor = None
        self.preprocessor_artifact = None
        self.metadata = None
        self.model_identifier = None

        self.app_logger.structured_log(
            logging.INFO,
            "ModelPredictor initialized",
            registry_type=type(model_registry).__name__
        )

    def load_model(self, model_identifier: str) -> None:
        """
        Load a model and its artifacts from the registry.

        Args:
            model_identifier: Model URI, registered name/version, or path
                            Examples:
                            - 'runs:/<run_id>/model'
                            - 'models:/model_name/Production'
                            - 'models:/model_name/1'

        Raises:
            Error if model cannot be loaded
        """
        try:
            self.app_logger.structured_log(
                logging.INFO,
                "Loading model for inference",
                model_identifier=model_identifier
            )

            # Load model and artifacts from registry
            model_data = self.model_registry.load_model(model_identifier)

            self.model = model_data['model']
            self.preprocessor_artifact = model_data.get('preprocessor_artifact')
            self.metadata = model_data.get('metadata', {})
            self.model_identifier = model_identifier

            # Restore preprocessor if artifact exists
            if self.preprocessor_artifact:
                # Create preprocessor instance and load fitted state
                self.preprocessor = Preprocessor(
                    self.config,
                    self.app_logger,
                    None,  # app_file_handler not needed for inference
                    self.error_handler
                )
                self.preprocessor.load_preprocessor_artifact(self.preprocessor_artifact)

                self.app_logger.structured_log(
                    logging.INFO,
                    "Preprocessor loaded from artifact",
                    model_name=self.preprocessor_artifact.get('model_name')
                )
            else:
                self.preprocessor = None
                self.app_logger.structured_log(
                    logging.INFO,
                    "No preprocessor artifact found - predictions will use raw features"
                )

            self.app_logger.structured_log(
                logging.INFO,
                "Model loaded successfully",
                model_identifier=model_identifier,
                has_preprocessor=self.preprocessor is not None
            )

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'inference',
                "Error loading model for inference",
                original_error=str(e),
                model_identifier=model_identifier
            )

    def predict(self, X: pd.DataFrame, return_probabilities: bool = True) -> np.ndarray:
        """
        Make predictions on new data.

        Automatically applies preprocessing if a preprocessor was loaded with the model.

        Args:
            X: Input features as DataFrame
            return_probabilities: If True, return probability estimates (for classification)
                                 If False, return class predictions

        Returns:
            Predictions as numpy array

        Raises:
            Error if model not loaded or prediction fails
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Call load_model() first.")

            self.app_logger.structured_log(
                logging.INFO,
                "Making predictions",
                input_shape=X.shape,
                return_probabilities=return_probabilities
            )

            # Apply preprocessing if preprocessor exists
            if self.preprocessor:
                self.app_logger.structured_log(
                    logging.INFO,
                    "Applying preprocessing to input data"
                )
                X_transformed = self.preprocessor.transform(X)

                self.app_logger.structured_log(
                    logging.INFO,
                    "Preprocessing applied",
                    transformed_shape=X_transformed.shape
                )
            else:
                X_transformed = X

            # Make predictions
            if return_probabilities:
                # Try to get probability predictions
                if hasattr(self.model, 'predict_proba'):
                    predictions = self.model.predict_proba(X_transformed)
                    # For binary classification, return probability of positive class
                    if predictions.shape[1] == 2:
                        predictions = predictions[:, 1]
                elif hasattr(self.model, 'predict'):
                    # MLflow pyfunc models use predict
                    predictions = self.model.predict(X_transformed)
                    # If predictions are 2D with 2 columns, extract positive class probability
                    if len(predictions.shape) > 1 and predictions.shape[1] == 2:
                        predictions = predictions[:, 1]
                else:
                    raise ValueError("Model does not support probability predictions")
            else:
                # Get class predictions
                if hasattr(self.model, 'predict'):
                    predictions = self.model.predict(X_transformed)
                else:
                    # If only probabilities available, threshold at 0.5
                    if hasattr(self.model, 'predict_proba'):
                        probs = self.model.predict_proba(X_transformed)
                        if probs.shape[1] == 2:
                            predictions = (probs[:, 1] >= 0.5).astype(int)
                        else:
                            predictions = np.argmax(probs, axis=1)
                    else:
                        raise ValueError("Model does not support predictions")

            self.app_logger.structured_log(
                logging.INFO,
                "Predictions generated",
                output_shape=predictions.shape,
                predictions_mean=float(np.mean(predictions))
            )

            return predictions

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'inference',
                "Error making predictions",
                original_error=str(e),
                input_shape=X.shape if X is not None else None
            )

    def predict_batch(self,
                     X: pd.DataFrame,
                     batch_size: int = 1000,
                     return_probabilities: bool = True) -> np.ndarray:
        """
        Make predictions on large datasets in batches.

        Useful for inference on large datasets that don't fit in memory.

        Args:
            X: Input features as DataFrame
            batch_size: Number of rows to process at once
            return_probabilities: If True, return probabilities; if False, return classes

        Returns:
            Predictions as numpy array
        """
        try:
            self.app_logger.structured_log(
                logging.INFO,
                "Starting batch prediction",
                input_shape=X.shape,
                batch_size=batch_size
            )

            n_samples = len(X)
            n_batches = (n_samples + batch_size - 1) // batch_size

            predictions_list = []

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)

                batch = X.iloc[start_idx:end_idx]
                batch_predictions = self.predict(batch, return_probabilities=return_probabilities)
                predictions_list.append(batch_predictions)

                self.app_logger.structured_log(
                    logging.DEBUG,
                    "Batch processed",
                    batch_number=i + 1,
                    total_batches=n_batches,
                    batch_size=len(batch)
                )

            predictions = np.concatenate(predictions_list)

            self.app_logger.structured_log(
                logging.INFO,
                "Batch prediction completed",
                total_predictions=len(predictions)
            )

            return predictions

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'inference',
                "Error in batch prediction",
                original_error=str(e),
                batch_size=batch_size
            )

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata and configuration
        """
        if self.model is None:
            return {'status': 'No model loaded'}

        info = {
            'model_identifier': self.model_identifier,
            'model_type': type(self.model).__name__,
            'has_preprocessor': self.preprocessor is not None,
            'metadata': self.metadata
        }

        if self.preprocessor:
            info['preprocessor_info'] = {
                'model_name': self.preprocessor_artifact.get('model_name'),
                'n_features_in': len(self.preprocessor_artifact.get('feature_names_in', [])),
                'n_features_out': len(self.preprocessor_artifact.get('feature_names_out', []))
            }

        return info

    def validate_input(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate input data against expected feature schema.

        Args:
            X: Input features to validate

        Returns:
            Dictionary with validation results:
            - 'valid': Boolean indicating if validation passed
            - 'missing_features': List of expected features not in input
            - 'extra_features': List of features in input not expected
            - 'feature_count': Expected vs actual feature count
        """
        if self.preprocessor is None:
            return {
                'valid': None,
                'message': 'No preprocessor loaded - cannot validate features'
            }

        expected_features = self.preprocessor_artifact.get('feature_names_in', [])
        actual_features = X.columns.tolist()

        missing_features = [f for f in expected_features if f not in actual_features]
        extra_features = [f for f in actual_features if f not in expected_features]

        valid = len(missing_features) == 0 and len(extra_features) == 0

        return {
            'valid': valid,
            'missing_features': missing_features,
            'extra_features': extra_features,
            'feature_count': {
                'expected': len(expected_features),
                'actual': len(actual_features)
            }
        }

    def is_loaded(self) -> bool:
        """
        Check if a model is currently loaded.

        Returns:
            True if model is loaded and ready for predictions
        """
        return self.model is not None
