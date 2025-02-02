import logging
import numpy as np
from typing import Dict
from sklearn.base import BaseEstimator
from .base_trainer import BaseTrainer
from ..data_classes import ModelTrainingResults
from ...logging.logging_utils import structured_log
from ...error_handling.custom_exceptions import ModelTestingError
from ...config.config import AbstractConfig

logger = logging.getLogger(__name__)

class SKLearnTrainer(BaseTrainer):
    def __init__(self, config: AbstractConfig):
        """Initialize SKLearn trainer with configuration."""
        super().__init__(config)
        self.model_registry = {
            'RandomForest': 'sklearn.ensemble.RandomForestRegressor',
            'ExtraTrees': 'sklearn.ensemble.ExtraTreesRegressor',
            'GradientBoosting': 'sklearn.ensemble.GradientBoostingRegressor',
            'LinearRegression': 'sklearn.linear_model.LinearRegression',
            'Ridge': 'sklearn.linear_model.Ridge',
            'Lasso': 'sklearn.linear_model.Lasso',
            'ElasticNet': 'sklearn.linear_model.ElasticNet',
            'SVR': 'sklearn.svm.SVR',
            # Add more models as needed
        }

    def train(self, X_train, y_train, X_val, y_val, fold: int, model_params: Dict, results: ModelTrainingResults) -> ModelTrainingResults:
        """Train a scikit-learn model with the enhanced ModelTrainingResults."""
        structured_log(logger, logging.INFO, "Starting SKLearn model training", 
                    input_shape=X_train.shape,
                    model_name=results.model_name)
        
        try:
            # Initialize the model
            model = self._initialize_model(results.model_name, model_params)

            # Train model
            model.fit(X_train, y_train)

            # Store model and generate predictions
            results.model = model
            results.predictions = model.predict(X_val)

            structured_log(logger, logging.INFO, "Generated predictions", 
                        predictions_shape=results.predictions.shape,
                        predictions_mean=float(np.mean(results.predictions)))

            # Calculate feature importance if available
            self._calculate_feature_importance(model, X_train, results)

            # Store feature data
            results.feature_data = X_val
            results.target_data = y_val
            
            return results

        except Exception as e:
            raise ModelTestingError(
                "Error in SKLearn model training",
                error_message=str(e),
                model_name=results.model_name,
                dataframe_shape=X_train.shape
            )

    def _initialize_model(self, model_name: str, model_params: Dict) -> BaseEstimator:
        """Initialize a scikit-learn model from the registry."""
        try:
            if model_name not in self.model_registry:
                raise ValueError(f"Unknown model: {model_name}")

            # Import the model class dynamically
            model_path = self.model_registry[model_name]
            module_path, class_name = model_path.rsplit('.', 1)
            
            # Dynamic import
            module = __import__(module_path, fromlist=[class_name])
            model_class = getattr(module, class_name)
            
            # Initialize model with parameters
            model = model_class(**model_params)
            
            return model

        except Exception as e:
            raise ModelTestingError(
                "Error initializing SKLearn model",
                error_message=str(e),
                model_name=model_name
            )

    def _calculate_feature_importance(self, model, X_train, results: ModelTrainingResults) -> None:
        """Calculate and store feature importance scores if available."""
        try:
            # Check for different types of feature importance attributes
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_scores = np.abs(model.coef_)
                if importance_scores.ndim > 1:
                    importance_scores = np.mean(importance_scores, axis=0)
            else:
                structured_log(logger, logging.WARNING, 
                            "Model does not provide feature importance scores",
                            model_type=type(model).__name__)
                importance_scores = np.zeros(X_train.shape[1])

            results.feature_importance_scores = importance_scores
            results.feature_names = X_train.columns.tolist()
            
            structured_log(logger, logging.INFO, "Calculated feature importance",
                        num_features_with_importance=np.sum(importance_scores > 0))
            
        except Exception as e:
            structured_log(logger, logging.WARNING, "Failed to calculate feature importance",
                        error=str(e))
            results.feature_importance_scores = np.zeros(X_train.shape[1])

    def _process_learning_curve_data(self, model, results: ModelTrainingResults) -> None:
        """
        Process learning curve data - not implemented for most sklearn models
        as they don't provide built-in learning curves.
        """
        pass

    def _calculate_shap_values(self, model, X_val, y_val, results: ModelTrainingResults) -> None:
        """
        Calculate SHAP values using the shap library if available.
        Note: This is implemented separately as it requires additional dependencies.
        """
        try:
            import shap
            
            # Choose explainer based on model type
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model) if hasattr(model, 'apply') else shap.KernelExplainer(model.predict_proba, X_val)
            else:
                explainer = shap.TreeExplainer(model) if hasattr(model, 'apply') else shap.KernelExplainer(model.predict, X_val)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_val)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values)
            if shap_values.ndim > 2:
                shap_values = shap_values.mean(axis=0)
                
            results.shap_values = shap_values
            structured_log(logger, logging.INFO, "Calculated SHAP values",
                        shap_values_shape=shap_values.shape)
            
        except ImportError:
            structured_log(logger, logging.WARNING, 
                        "SHAP library not available. SHAP values not calculated.")
        except Exception as e:
            structured_log(logger, logging.ERROR, "Failed to calculate SHAP values",
                        error=str(e),
                        error_type=type(e).__name__)

    def _generate_learning_curve_data(self, X_train, y_train, X_val, y_val, 
                                    fold: int, model_params: Dict,
                                    results: ModelTrainingResults) -> ModelTrainingResults:
        """Generate learning curve data for sklearn models."""
        try:
            model = self._initialize_model(results.model_name, model_params)
            model.fit(X_train, y_train)
            
            # Get predictions
            train_pred = (model.predict_proba(X_train)[:, 1] 
                        if hasattr(model, 'predict_proba') 
                        else model.predict(X_train))
            val_pred = (model.predict_proba(X_val)[:, 1]
                    if hasattr(model, 'predict_proba')
                    else model.predict(X_val))

            # Calculate scores
            train_score = self._calculate_model_score(y_train, train_pred)
            val_score = self._calculate_model_score(y_val, val_pred)

            # For sklearn models, we only have one iteration (the final model)
            results.learning_curve_data.add_iteration(
                train_score=train_score,
                val_score=val_score,
                iteration=0
            )
            results.learning_curve_data.metric_name = 'accuracy'  # or whatever metric you're using

            return results

        except Exception as e:
            structured_log(logger, logging.ERROR,
                        "Error in sklearn learning curve generation",
                        error=str(e))
            raise

    def _calculate_model_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate model score based on prediction type."""
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:  # Multi-class probabilities
            y_pred = np.argmax(y_pred, axis=1)
        elif y_pred.dtype == np.float64:  # Binary probabilities
            y_pred = (y_pred >= 0.5).astype(int)
        return np.mean(y_true == y_pred) 