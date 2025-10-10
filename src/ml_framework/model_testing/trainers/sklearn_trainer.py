import logging
import numpy as np
from typing import Dict, Optional, Any, Tuple
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from .base_trainer import BaseTrainer
from .trainer_utils import TrainerUtils
from ml_framework.framework.data_classes import ModelTrainingResults
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler

class SKLearnTrainer(BaseTrainer):
    def __init__(self, 
                config: BaseConfigManager, 
                app_logger: BaseAppLogger, 
                error_handler: BaseErrorHandler,
                model_type: Optional[str] = None):
        """
        Initialize SKLearn trainer with configuration and dependencies.
        
        Args:
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler
            model_type: Type of sklearn model (e.g., 'randomforest', 'logisticregression')
        """
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler
        self.utils = TrainerUtils(app_logger, error_handler)
        self.model_type = model_type
        self._model_cfg = config.core.model_testing_config
        
        self.model_registry = {
            'randomforest': RandomForestClassifier,
            'logisticregression': LogisticRegression,
            'histgradientboosting': HistGradientBoostingClassifier
        }
        
        self.app_logger.structured_log(
            logging.INFO, 
            "SKLearnTrainer initialized successfully",
            trainer_type=type(self).__name__,
            model_type=model_type
        )

    @staticmethod
    def log_performance(func):
        """Decorator factory for performance logging"""
        def wrapper(*args, **kwargs):
            # Get the self instance from args since this is now a static method
            instance = args[0]
            return instance.app_logger.log_performance(func)(*args, **kwargs)
        return wrapper                                                                  

    def _initialize_model(self, model_name: str, model_params: Dict) -> BaseEstimator:
        """Initialize a scikit-learn model with proper error handling."""
        try:
            if model_name not in self.model_registry:
                raise ValueError(f"Unknown model: {model_name}")

            model_class = self.model_registry[model_name]
            
            # Handle special parameters for different model types
            if model_name == 'HistGradientBoosting':
                if 'categorical_features' in model_params:
                    model_params = model_params.copy()
                    cat_features = model_params.pop('categorical_features')
                    if cat_features:
                        model_params['categorical_features'] = [
                            i for i, col in enumerate(self.feature_names) 
                            if col in cat_features
                        ]
            
            return model_class(**model_params)

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'model_testing',
                "Error initializing SKLearn model",
                error_message=str(e),
                model_name=model_name
            )

    @log_performance
    def train(self, X_train, y_train, X_val, y_val, fold: int, model_params: Dict, results: ModelTrainingResults) -> ModelTrainingResults:
        """Train a scikit-learn model with enhanced tracking and evaluation."""
        if self.model_type:
            results.model_name = self.model_type
        
        self.app_logger.structured_log(
            logging.INFO, 
            "Starting SKLearn model training", 
            input_shape=X_train.shape,
            model_name=results.model_name
        )
        
        try:
            # Store feature names for categorical feature handling
            self.feature_names = X_train.columns.tolist()
            
            # Extract eval_metric before initializing model
            eval_metric = model_params.pop('eval_metric', 'accuracy')
            
            # Initialize the model with remaining params
            model = self._initialize_model(results.model_name, model_params)
            
            # Store configuration including eval_metric
            results.model_params = model_params
            results.eval_metric = eval_metric
            results.feature_names = self.feature_names
            
            # Handle categorical features if specified
            if hasattr(self._model_cfg, 'categorical_features'):
                results.categorical_features = self._model_cfg.categorical_features
            
            # Train model
            model.fit(X_train, y_train)
            results.model = model
            
            # Generate predictions
            results.probability_predictions = self._get_probability_predictions(model, X_val)
            results.predictions = results.probability_predictions
            
            self.app_logger.structured_log(
                logging.INFO, 
                "Generated predictions", 
                predictions_shape=results.predictions.shape,
                predictions_mean=float(np.mean(results.predictions))
            )
            
            # Store feature data
            results.feature_data = X_val
            results.target_data = y_val
            
            # Calculate feature importance
            self._calculate_feature_importance(model, X_train, results)
            
            # Calculate SHAP values if configured
            if hasattr(self._model_cfg, 'calculate_shap_values') and self._model_cfg.calculate_shap_values:
                self._calculate_shap_values(model, X_val, y_val, results)

                if hasattr(self._model_cfg, 'calculate_shap_interactions') and self._model_cfg.calculate_shap_interactions:
                    self._calculate_shap_interactions(model, X_val, results)
            
            return results

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'model_testing',
                "Error in SKLearn model training",
                error_message=str(e),
                model_name=results.model_name,
                dataframe_shape=X_train.shape
            )

    def _get_probability_predictions(self, model: BaseEstimator, X: Any) -> np.ndarray:
        """Get probability predictions with consistent output."""
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            return probs[:, 1] if probs.shape[1] == 2 else probs
        return model.predict(X)

    def _calculate_feature_importance(self, model, X_train, results: ModelTrainingResults) -> None:
        """Calculate and store feature importance scores with proper handling for different model types."""
        try:
            # Different models expose feature importances in different ways
            if isinstance(model, HistGradientBoostingClassifier):
                # For HistGradientBoosting, use permutation importance
                r = permutation_importance(
                    model, X_train, results.target_data,
                    n_repeats=5,
                    random_state=self.config.random_state if hasattr(self.config, 'random_state') else 42,
                    n_jobs=-1
                )
                importance_scores = r.importances_mean
                
            elif hasattr(model, 'feature_importances_'):  # Tree-based models
                importance_scores = model.feature_importances_
            elif hasattr(model, 'coef_'):  # Linear models
                importance_scores = np.abs(model.coef_)
                if importance_scores.ndim > 1:
                    importance_scores = np.mean(importance_scores, axis=0)
            else:
                self.app_logger.structured_log(
                    logging.WARNING, 
                    "Model does not provide feature importance scores",
                    model_type=type(model).__name__
                )
                importance_scores = np.zeros(X_train.shape[1])

            # Normalize importance scores to [0,1] range
            if len(importance_scores) > 0 and importance_scores.max() > 0:
                importance_scores = importance_scores / importance_scores.max()

            results.feature_importance_scores = importance_scores
            
            self.app_logger.structured_log(
                logging.INFO, 
                "Calculated feature importance",
                num_features_with_importance=np.sum(importance_scores > 0),
                max_importance=float(np.max(importance_scores)),
                min_importance=float(np.min(importance_scores))
            )
            
        except Exception as e:
            self.app_logger.structured_log(
                logging.WARNING, 
                "Failed to calculate feature importance",
                error=str(e),
                model_type=type(model).__name__
            )
            results.feature_importance_scores = np.zeros(X_train.shape[1])

    def _calculate_shap_values(self, model, X_val, y_val, results: ModelTrainingResults) -> None:
        """Calculate SHAP values with memory and model-type considerations."""
        try:
            import shap
            
            # Choose appropriate explainer based on model type
            if isinstance(model, (RandomForestClassifier, HistGradientBoostingClassifier)):
                explainer = shap.TreeExplainer(model)
            else:  # For LogisticRegression and other models
                background = shap.sample(X_val, min(100, len(X_val)))
                predict_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
                explainer = shap.KernelExplainer(predict_fn, background)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_val)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            if shap_values.ndim > 2:
                shap_values = shap_values.mean(axis=0)
                
            results.shap_values = shap_values
            
            self.app_logger.structured_log(
                logging.INFO, 
                "Calculated SHAP values",
                shap_values_shape=shap_values.shape
            )
            
        except ImportError:
            self.app_logger.structured_log(
                logging.WARNING, 
                "SHAP library not available. SHAP values not calculated."
            )
        except Exception as e:
            self.app_logger.structured_log(
                logging.ERROR, 
                "Failed to calculate SHAP values",
                error=str(e)
            )

    def _calculate_shap_interactions(self, model, X_val, results: ModelTrainingResults) -> None:
        """Calculate SHAP interaction values if supported by the model."""
        try:
            if not isinstance(model, (RandomForestClassifier, HistGradientBoostingClassifier)):
                self.app_logger.structured_log(
                    logging.INFO, 
                    "SHAP interactions not supported for this model type"
                )
                return

            import shap
            n_features = X_val.shape[1]
            estimated_memory = (X_val.shape[0] * n_features * n_features * 8) / (1024 ** 3)

            if not hasattr(self._model_cfg, 'max_shap_interaction_memory_gb') or estimated_memory <= self._model_cfg.max_shap_interaction_memory_gb:
                explainer = shap.TreeExplainer(model)
                interaction_values = explainer.shap_interaction_values(X_val)
                
                if isinstance(interaction_values, list):
                    interaction_values = interaction_values[1]
                    
                results.shap_interaction_values = interaction_values
                self.app_logger.structured_log(
                    logging.INFO, 
                    "Calculated SHAP interaction values",
                    interaction_shape=interaction_values.shape
                )
            else:
                self.app_logger.structured_log(
                    logging.WARNING, 
                    "Skipping SHAP interactions due to memory constraints",
                    estimated_memory_gb=estimated_memory
                )
                
        except Exception as e:
            self.app_logger.structured_log(
                logging.ERROR, 
                "Failed to calculate SHAP interactions",
                error=str(e)
            )
            
    def _convert_metric_scores(self, train_score: float, val_score: float, metric_name: str) -> Tuple[float, float]:
        """Convert metric scores to a consistent format (higher is better)."""
        return self.utils._convert_metric_scores(train_score, val_score, metric_name)
        
    def _process_learning_curve_data(self, evals_result: Dict, results: ModelTrainingResults) -> None:
        """Process and store learning curve data."""
        return self.utils._process_learning_curve_data(evals_result, results)