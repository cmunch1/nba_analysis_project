"""
Abstract base class for model registry implementations.

This provides a consistent interface for saving, loading, and managing trained models
with their associated artifacts (preprocessors, configs, metadata).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler


class BaseModelRegistry(ABC):
    """
    Abstract base class for model registry implementations.

    Supports multiple backends (MLflow, custom file-based, cloud services, etc.)
    while providing a consistent interface for model persistence and retrieval.
    """

    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        """
        Initialize model registry with configuration and dependencies.

        Args:
            config: Configuration manager
            app_logger: Application logger for structured logging
            error_handler: Error handler for standardized error management
        """
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler

    @abstractmethod
    def save_model(self,
                   model: Any,
                   model_name: str,
                   preprocessor_artifact: Optional[Dict[str, Any]] = None,
                   signature: Optional[Any] = None,
                   input_example: Optional[pd.DataFrame] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   tags: Optional[Dict[str, str]] = None) -> str:
        """
        Save a trained model with its artifacts to the registry.

        Args:
            model: Trained model object
            model_name: Name to register the model under
            preprocessor_artifact: Optional fitted preprocessor for inference
            signature: Optional model signature (input/output schema)
            input_example: Optional example input for validation
            metadata: Optional metadata (hyperparameters, metrics, etc.)
            tags: Optional tags for categorization and filtering

        Returns:
            Model identifier (path, URI, or version string)
        """
        pass

    @abstractmethod
    def load_model(self, model_identifier: str) -> Dict[str, Any]:
        """
        Load a trained model and its artifacts from the registry.

        Args:
            model_identifier: Identifier to locate the model (path, URI, version, etc.)

        Returns:
            Dictionary containing:
                - 'model': The loaded model object
                - 'preprocessor_artifact': Fitted preprocessor (if available)
                - 'metadata': Model metadata (if available)
                - 'signature': Model signature (if available)
        """
        pass

    @abstractmethod
    def list_models(self,
                    model_name: Optional[str] = None,
                    tags: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        List available models in the registry.

        Args:
            model_name: Optional filter by model name
            tags: Optional filter by tags

        Returns:
            List of model metadata dictionaries
        """
        pass

    @abstractmethod
    def delete_model(self, model_identifier: str) -> bool:
        """
        Delete a model from the registry.

        Args:
            model_identifier: Identifier of the model to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        pass

    @abstractmethod
    def get_model_metadata(self, model_identifier: str) -> Dict[str, Any]:
        """
        Get metadata for a specific model.

        Args:
            model_identifier: Identifier of the model

        Returns:
            Dictionary containing model metadata
        """
        pass

    @abstractmethod
    def register_model_version(self,
                              model_identifier: str,
                              version: str,
                              stage: Optional[str] = None,
                              description: Optional[str] = None) -> bool:
        """
        Register a specific version of a model (e.g., for production deployment).

        Args:
            model_identifier: Identifier of the model
            version: Version string or number
            stage: Optional stage (e.g., 'staging', 'production', 'archived')
            description: Optional description of this version

        Returns:
            True if registration was successful, False otherwise
        """
        pass

    @abstractmethod
    def transition_model_stage(self,
                              model_identifier: str,
                              stage: str) -> bool:
        """
        Transition a model to a different stage (e.g., staging -> production).

        Args:
            model_identifier: Identifier of the model
            stage: Target stage name

        Returns:
            True if transition was successful, False otherwise
        """
        pass

    @abstractmethod
    def get_model_by_stage(self,
                          model_name: str,
                          stage: str) -> Optional[Dict[str, Any]]:
        """
        Load the latest model in a specific stage.

        Args:
            model_name: Name of the model
            stage: Stage to retrieve from (e.g., 'production')

        Returns:
            Dictionary containing model and artifacts, or None if not found
        """
        pass
