"""
MLflow implementation of the model registry interface.

Provides model persistence and retrieval using MLflow Model Registry.
"""

import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from typing import Dict, Any, Optional, List
import pandas as pd
import pickle
import os
from pathlib import Path

from .base_model_registry import BaseModelRegistry
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler


class MLflowModelRegistry(BaseModelRegistry):
    """
    MLflow implementation of model registry.

    Uses MLflow Model Registry for versioning, staging, and deployment workflows.
    Stores preprocessor artifacts as additional files alongside models.
    """

    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        """
        Initialize MLflow model registry.

        Args:
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler
        """
        super().__init__(config, app_logger, error_handler)

        # Set tracking URI
        tracking_uri = getattr(config, 'tracking_uri', None)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            local_mlruns_path = f"file://{os.path.abspath('mlruns')}"
            mlflow.set_tracking_uri(local_mlruns_path)

        self.app_logger.structured_log(
            logging.INFO,
            "MLflow Model Registry initialized",
            tracking_uri=mlflow.get_tracking_uri()
        )

    def save_model(self,
                   model: Any,
                   model_name: str,
                   preprocessor_artifact: Optional[Dict[str, Any]] = None,
                   signature: Optional[Any] = None,
                   input_example: Optional[pd.DataFrame] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   tags: Optional[Dict[str, str]] = None,
                   additional_artifacts: Optional[Dict[str, Any]] = None) -> str:
        """
        Save model to MLflow registry with artifacts.

        Args:
            model: Trained model object
            model_name: Name to register the model under
            preprocessor_artifact: Optional fitted preprocessor
            signature: Optional MLflow signature
            input_example: Optional example input
            metadata: Optional metadata dict
            tags: Optional tags dict
            additional_artifacts: Optional dict of additional artifacts (e.g., calibrator, conformal_predictor)

        Returns:
            Model URI (e.g., 'runs:/<run_id>/<artifact_path>')
        """
        try:
            self.app_logger.structured_log(
                logging.INFO,
                "Saving model to MLflow registry",
                model_name=model_name,
                has_preprocessor=preprocessor_artifact is not None
            )

            # Determine model flavor based on model type
            model_type = type(model).__name__

            # Start or use existing MLflow run
            active_run = mlflow.active_run()
            if active_run is None:
                run = mlflow.start_run()
            else:
                run = active_run

            try:
                # Log metadata as parameters
                if metadata:
                    for key, value in metadata.items():
                        # MLflow params must be strings
                        if isinstance(value, (dict, list)):
                            mlflow.log_param(key, str(value))
                        else:
                            mlflow.log_param(key, value)

                # Log tags
                if tags:
                    mlflow.set_tags(tags)

                # Save preprocessor artifact if provided
                if preprocessor_artifact:
                    preprocessor_path = "preprocessor_artifact.pkl"
                    with open(preprocessor_path, 'wb') as f:
                        pickle.dump(preprocessor_artifact, f)
                    mlflow.log_artifact(preprocessor_path, artifact_path="artifacts")
                    os.remove(preprocessor_path)  # Clean up local file

                    self.app_logger.structured_log(
                        logging.INFO,
                        "Preprocessor artifact saved",
                        artifact_path="artifacts/preprocessor_artifact.pkl"
                    )

                    # Save explicit feature schema for inference-time validation and reordering
                    if 'feature_names_in' in preprocessor_artifact:
                        feature_schema = {
                            'feature_names': list(preprocessor_artifact['feature_names_in']),
                            'num_features': len(preprocessor_artifact['feature_names_in']),
                            'schema_version': '1.0'
                        }
                        schema_path = "feature_schema.pkl"
                        with open(schema_path, 'wb') as f:
                            pickle.dump(feature_schema, f)
                        mlflow.log_artifact(schema_path, artifact_path="artifacts")
                        os.remove(schema_path)

                        self.app_logger.structured_log(
                            logging.INFO,
                            "Feature schema artifact saved",
                            num_features=feature_schema['num_features'],
                            artifact_path="artifacts/feature_schema.pkl"
                        )

                # Save additional artifacts if provided (e.g., calibrator, conformal_predictor)
                if additional_artifacts:
                    for artifact_name, artifact_data in additional_artifacts.items():
                        artifact_path = f"{artifact_name}_artifact.pkl"
                        with open(artifact_path, 'wb') as f:
                            pickle.dump(artifact_data, f)
                        mlflow.log_artifact(artifact_path, artifact_path="artifacts")
                        os.remove(artifact_path)  # Clean up local file

                        self.app_logger.structured_log(
                            logging.INFO,
                            "Additional artifact saved",
                            artifact_name=artifact_name,
                            artifact_path=f"artifacts/{artifact_name}_artifact.pkl"
                        )

                # Log the model using appropriate MLflow flavor
                if 'XGBoost' in model_type or 'Booster' in model_type:
                    mlflow.xgboost.log_model(
                        model,
                        artifact_path="model",
                        signature=signature,
                        input_example=input_example,
                        registered_model_name=model_name
                    )
                elif 'LightGBM' in model_type or 'LGBMClassifier' in model_type:
                    mlflow.lightgbm.log_model(
                        model,
                        artifact_path="model",
                        signature=signature,
                        input_example=input_example,
                        registered_model_name=model_name
                    )
                elif 'CatBoost' in model_type:
                    mlflow.catboost.log_model(
                        model,
                        artifact_path="model",
                        signature=signature,
                        input_example=input_example,
                        registered_model_name=model_name
                    )
                elif 'torch' in str(type(model).__module__):
                    mlflow.pytorch.log_model(
                        model,
                        artifact_path="model",
                        signature=signature,
                        input_example=input_example,
                        registered_model_name=model_name
                    )
                else:
                    # Default to sklearn flavor for scikit-learn models
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path="model",
                        signature=signature,
                        input_example=input_example,
                        registered_model_name=model_name
                    )

                model_uri = f"runs:/{run.info.run_id}/model"

                self.app_logger.structured_log(
                    logging.INFO,
                    "Model saved to MLflow registry",
                    model_name=model_name,
                    model_uri=model_uri,
                    run_id=run.info.run_id
                )

                return model_uri

            finally:
                # Only end run if we started it
                if active_run is None:
                    mlflow.end_run()

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'model_registry',
                "Error saving model to MLflow registry",
                original_error=str(e),
                model_name=model_name
            )

    def load_model(self, model_identifier: str) -> Dict[str, Any]:
        """
        Load model and artifacts from MLflow registry.

        Args:
            model_identifier: Model URI or registered model name with version
                            (e.g., 'runs:/<run_id>/model' or 'models:/model_name/1')

        Returns:
            Dictionary with 'model', 'preprocessor_artifact', 'metadata'
        """
        try:
            self.app_logger.structured_log(
                logging.INFO,
                "Loading model from MLflow registry",
                model_identifier=model_identifier
            )

            # Load the model
            model = mlflow.pyfunc.load_model(model_identifier)

            # Extract run_id from model URI
            run_id = None
            if model_identifier.startswith('runs:/'):
                run_id = model_identifier.split('/')[1]
            elif model_identifier.startswith('models:/'):
                # Get run_id from registered model version or stage
                model_name = model_identifier.split('/')[1]
                version_or_stage = model_identifier.split('/')[-1]
                client = mlflow.tracking.MlflowClient()

                # Check if version_or_stage is a stage name (Production, Staging, Archived) or a version number
                if version_or_stage.isdigit():
                    # It's a version number
                    model_version = client.get_model_version(model_name, version_or_stage)
                    run_id = model_version.run_id
                else:
                    # It's a stage name - search for models in that stage
                    versions = client.search_model_versions(f"name='{model_name}'")
                    stage_versions = [v for v in versions if v.current_stage == version_or_stage]
                    if not stage_versions:
                        raise ValueError(f"No model version found in stage '{version_or_stage}' for model '{model_name}'")
                    # Use the first (most recent) version in this stage
                    model_version = stage_versions[0]
                    run_id = model_version.run_id
                    self.app_logger.structured_log(
                        logging.INFO,
                        f"Using model version from stage",
                        model_name=model_name,
                        stage=version_or_stage,
                        version=model_version.version,
                        run_id=run_id
                    )

            # Load preprocessor artifact if available
            preprocessor_artifact = None
            if run_id:
                try:
                    artifact_path = "artifacts/preprocessor_artifact.pkl"
                    local_path = mlflow.artifacts.download_artifacts(
                        artifact_uri=f"runs:/{run_id}/{artifact_path}"
                    )
                    with open(local_path, 'rb') as f:
                        preprocessor_artifact = pickle.load(f)

                    self.app_logger.structured_log(
                        logging.INFO,
                        "Preprocessor artifact loaded",
                        run_id=run_id
                    )
                except Exception as e:
                    self.app_logger.structured_log(
                        logging.WARNING,
                        "No preprocessor artifact found for model",
                        run_id=run_id,
                        error=str(e)
                    )

            # Load feature schema artifact if available
            feature_schema = None
            if run_id:
                try:
                    artifact_path = "artifacts/feature_schema.pkl"
                    local_path = mlflow.artifacts.download_artifacts(
                        artifact_uri=f"runs:/{run_id}/{artifact_path}"
                    )
                    with open(local_path, 'rb') as f:
                        feature_schema = pickle.load(f)

                    self.app_logger.structured_log(
                        logging.INFO,
                        "Feature schema artifact loaded",
                        num_features=feature_schema.get('num_features', 'unknown'),
                        run_id=run_id
                    )
                except Exception as e:
                    self.app_logger.structured_log(
                        logging.WARNING,
                        "No feature schema artifact found for model",
                        run_id=run_id,
                        error=str(e)
                    )

            # Load additional artifacts (calibrator, conformal_predictor, etc.)
            additional_artifacts = {}
            if run_id:
                for artifact_name in ['calibrator', 'conformal_predictor']:
                    try:
                        artifact_path = f"artifacts/{artifact_name}_artifact.pkl"
                        local_path = mlflow.artifacts.download_artifacts(
                            artifact_uri=f"runs:/{run_id}/{artifact_path}"
                        )
                        with open(local_path, 'rb') as f:
                            additional_artifacts[artifact_name] = pickle.load(f)

                        self.app_logger.structured_log(
                            logging.INFO,
                            f"{artifact_name.capitalize()} artifact loaded",
                            run_id=run_id
                        )
                    except Exception as e:
                        self.app_logger.structured_log(
                            logging.DEBUG,
                            f"No {artifact_name} artifact found for model",
                            run_id=run_id,
                            error=str(e)
                        )

            # Load metadata
            metadata = {}
            if run_id:
                try:
                    client = mlflow.tracking.MlflowClient()
                    run = client.get_run(run_id)
                    metadata = {
                        'params': run.data.params,
                        'metrics': run.data.metrics,
                        'tags': run.data.tags
                    }
                except Exception as e:
                    self.app_logger.structured_log(
                        logging.WARNING,
                        "Could not load metadata",
                        error=str(e)
                    )

            result = {
                'model': model,
                'preprocessor_artifact': preprocessor_artifact,
                'feature_schema': feature_schema,
                'metadata': metadata,
                'run_id': run_id,
                **additional_artifacts  # Include calibrator and conformal_predictor if they exist
            }

            self.app_logger.structured_log(
                logging.INFO,
                "Model loaded successfully",
                model_identifier=model_identifier,
                has_preprocessor=preprocessor_artifact is not None,
                has_calibrator='calibrator' in additional_artifacts,
                has_conformal_predictor='conformal_predictor' in additional_artifacts
            )

            return result

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'model_registry',
                "Error loading model from MLflow registry",
                original_error=str(e),
                model_identifier=model_identifier
            )

    def list_models(self,
                    model_name: Optional[str] = None,
                    tags: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        List registered models in MLflow.

        Args:
            model_name: Optional filter by model name
            tags: Optional filter by tags (not fully supported in MLflow yet)

        Returns:
            List of model metadata dictionaries
        """
        try:
            client = mlflow.tracking.MlflowClient()

            if model_name:
                # Get specific model versions
                versions = client.search_model_versions(f"name='{model_name}'")
                return [
                    {
                        'name': v.name,
                        'version': v.version,
                        'stage': v.current_stage,
                        'run_id': v.run_id,
                        'status': v.status
                    }
                    for v in versions
                ]
            else:
                # List all registered models
                models = client.search_registered_models()
                return [
                    {
                        'name': m.name,
                        'latest_versions': [
                            {
                                'version': v.version,
                                'stage': v.current_stage,
                                'run_id': v.run_id
                            }
                            for v in m.latest_versions
                        ]
                    }
                    for m in models
                ]

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'model_registry',
                "Error listing models",
                original_error=str(e)
            )

    def delete_model(self, model_identifier: str) -> bool:
        """
        Delete a registered model version.

        Args:
            model_identifier: Format 'model_name/version'

        Returns:
            True if successful
        """
        try:
            parts = model_identifier.split('/')
            if len(parts) != 2:
                raise ValueError("Model identifier must be in format 'model_name/version'")

            model_name, version = parts
            client = mlflow.tracking.MlflowClient()
            client.delete_model_version(model_name, version)

            self.app_logger.structured_log(
                logging.INFO,
                "Model version deleted",
                model_name=model_name,
                version=version
            )
            return True

        except Exception as e:
            self.app_logger.structured_log(
                logging.ERROR,
                "Error deleting model",
                error=str(e),
                model_identifier=model_identifier
            )
            return False

    def get_model_metadata(self, model_identifier: str) -> Dict[str, Any]:
        """
        Get metadata for a model version.

        Args:
            model_identifier: Format 'model_name/version'

        Returns:
            Dictionary with metadata
        """
        try:
            parts = model_identifier.split('/')
            if len(parts) != 2:
                raise ValueError("Model identifier must be in format 'model_name/version'")

            model_name, version = parts
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_model_version(model_name, version)

            return {
                'name': model_version.name,
                'version': model_version.version,
                'stage': model_version.current_stage,
                'status': model_version.status,
                'run_id': model_version.run_id,
                'description': model_version.description,
                'tags': model_version.tags
            }

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'model_registry',
                "Error getting model metadata",
                original_error=str(e),
                model_identifier=model_identifier
            )

    def register_model_version(self,
                              model_identifier: str,
                              version: str,
                              stage: Optional[str] = None,
                              description: Optional[str] = None) -> bool:
        """
        Register a model version with optional stage and description.

        Args:
            model_identifier: Run URI (e.g., 'runs:/<run_id>/model')
            version: Version string
            stage: Optional stage
            description: Optional description

        Returns:
            True if successful
        """
        try:
            # This is handled automatically by log_model with registered_model_name
            # This method is for updating existing versions

            # Extract model name from identifier if needed
            # For now, log a message that registration happens during save
            self.app_logger.structured_log(
                logging.INFO,
                "Model version registration happens during save_model",
                model_identifier=model_identifier
            )
            return True

        except Exception as e:
            self.app_logger.structured_log(
                logging.ERROR,
                "Error registering model version",
                error=str(e)
            )
            return False

    def transition_model_stage(self,
                              model_identifier: str,
                              stage: str) -> bool:
        """
        Transition a model version to a different stage.

        Args:
            model_identifier: Format 'model_name/version'
            stage: Target stage ('Staging', 'Production', 'Archived', or 'None')

        Returns:
            True if successful
        """
        try:
            parts = model_identifier.split('/')
            if len(parts) != 2:
                raise ValueError("Model identifier must be in format 'model_name/version'")

            model_name, version = parts
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )

            self.app_logger.structured_log(
                logging.INFO,
                "Model stage transitioned",
                model_name=model_name,
                version=version,
                stage=stage
            )
            return True

        except Exception as e:
            self.app_logger.structured_log(
                logging.ERROR,
                "Error transitioning model stage",
                error=str(e),
                model_identifier=model_identifier
            )
            return False

    def get_model_by_stage(self,
                          model_name: str,
                          stage: str) -> Optional[Dict[str, Any]]:
        """
        Load the latest model in a specific stage.

        Args:
            model_name: Name of the registered model
            stage: Stage to retrieve ('Staging', 'Production', etc.)

        Returns:
            Dictionary with model and artifacts, or None if not found
        """
        try:
            model_uri = f"models:/{model_name}/{stage}"
            return self.load_model(model_uri)

        except Exception as e:
            self.app_logger.structured_log(
                logging.WARNING,
                "Could not load model by stage",
                model_name=model_name,
                stage=stage,
                error=str(e)
            )
            return None
