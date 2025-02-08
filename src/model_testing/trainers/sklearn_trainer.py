import logging
import numpy as np
from typing import Dict, Optional, Any
from sklearn.base import BaseEstimator
from .base_trainer import BaseTrainer
from ..data_classes import ModelTrainingResults, LearningCurveData
from ...logging.logging_utils import structured_log
from ...error_handling.custom_exceptions import ModelTestingError
from ...config.config import AbstractConfig

logger = logging.getLogger(__name__)

class SKLearnTrainer(BaseTrainer):
    def __init__(self, config: AbstractConfig, model_type: str = None):
        """Initialize SKLearn trainer with configuration."""
        super().__init__(config)
        self.model_type = model_type
        self.model_registry = {
            'RandomForest': 'sklearn.ensemble.RandomForestClassifier',
            'LogisticRegression': 'sklearn.linear_model.LogisticRegression',
            'HistGradientBoosting': 'sklearn.ensemble.HistGradientBoostingClassifier'
        }

    def train(self, X_train, y_train, X_val, y_val, fold: int, 
              model_params: Dict, results: ModelTrainingResults) -> ModelTrainingResults:
        """Train a scikit-learn model with enhanced tracking and evaluation."""
        if self.model_type:
            results.model_name = self.model_type
        
        structured_log(logger, logging.INFO, "Starting SKLearn model training", 
                      input_shape=X_train.shape,
                      model_name=results.model_name)
        
        try:
            # Extract eval_metric before initializing model
            eval_metric = model_params.pop('eval_metric', 'accuracy')
            
            # Initialize the model with remaining params
            model = self._initialize_model(results.model_name, model_params)
            
            # Store configuration including eval_metric
            results.model_params = model_params
            results.eval_metric = eval_metric  # Store separately
            results.feature_names = X_train.columns.tolist()
            
            # Handle categorical features if specified
            if self.config.categorical_features:
                results.categorical_features = self.config.categorical_features
                
            # Generate learning curves if configured
            if self.config.generate_learning_curve_data:
                results = self._generate_learning_curve_data(
                    X_train, y_train, X_val, y_val, 
                    fold, model_params, results
                )
            
            # Train final model
            model.fit(X_train, y_train)
            results.model = model
            
            # Generate and store predictions
            results.probability_predictions = self._get_probability_predictions(model, X_val)
            results.predictions = results.probability_predictions
            
            structured_log(logger, logging.INFO, "Generated predictions", 
                         predictions_shape=results.predictions.shape,
                         predictions_mean=float(np.mean(results.predictions)))
            
            # Store feature data
            results.feature_data = X_val
            results.target_data = y_val
            
            # Calculate feature importance
            self._calculate_feature_importance(model, X_train, results)
            
            # Calculate SHAP values if configured
            if self.config.calculate_shap_values:
                self._calculate_shap_values(model, X_val, y_val, results)
                
                if self.config.calculate_shap_interactions:
                    self._calculate_shap_interactions(model, X_val, results)
            
            return results

        except Exception as e:
            raise ModelTestingError(
                "Error in SKLearn model training",
                error_message=str(e),
                model_name=results.model_name,
                dataframe_shape=X_train.shape
            )

    def _initialize_model(self, model_name: str, model_params: Dict) -> BaseEstimator:
        """Initialize a scikit-learn model with proper error handling."""
        try:
            if model_name not in self.model_registry:
                raise ValueError(f"Unknown model: {model_name}")

            module_path, class_name = self.model_registry[model_name].rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            model_class = getattr(module, class_name)
            
            # Handle special parameters for different model types
            if model_name == 'HistGradientBoosting':
                if 'categorical_features' in model_params:
                    model_params = model_params.copy()
                    cat_features = model_params.pop('categorical_features')
                    if cat_features:
                        model_params['categorical_features'] = [
                            i for i, col in enumerate(X_train.columns) 
                            if col in cat_features
                        ]
            
            return model_class(**model_params)

        except Exception as e:
            raise ModelTestingError(
                "Error initializing SKLearn model",
                error_message=str(e),
                model_name=model_name
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
            if hasattr(model, 'feature_importances_'):  # Tree-based models
                importance_scores = model.feature_importances_
            elif hasattr(model, 'coef_'):  # Linear models
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

    def _calculate_shap_values(self, model, X_val, y_val, results: ModelTrainingResults) -> None:
        """Calculate SHAP values with memory and model-type considerations."""
        try:
            import shap
            
            # Choose appropriate explainer based on model type
            if isinstance(model, (RandomForestClassifier, HistGradientBoostingClassifier)):
                explainer = shap.TreeExplainer(model)
            else:  # For LogisticRegression and other models
                if hasattr(model, 'predict_proba'):
                    background = shap.sample(X_val, 100)  # Sample background data
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                else:
                    explainer = shap.KernelExplainer(model.predict, shap.sample(X_val, 100))
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_val)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values)
            if shap_values.ndim > 2:  # For multi-class output
                shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values.mean(axis=0)
                
            results.shap_values = shap_values
            
            structured_log(logger, logging.INFO, "Calculated SHAP values",
                         shap_values_shape=shap_values.shape)
            
        except ImportError:
            structured_log(logger, logging.WARNING, 
                         "SHAP library not available. SHAP values not calculated.")
        except Exception as e:
            structured_log(logger, logging.ERROR, "Failed to calculate SHAP values",
                         error=str(e))

    def _calculate_shap_interactions(self, model, X_val, results: ModelTrainingResults) -> None:
        """Calculate SHAP interaction values if supported by the model."""
        try:
            if not isinstance(model, (RandomForestClassifier, HistGradientBoostingClassifier)):
                structured_log(logger, logging.INFO, 
                             "SHAP interactions not supported for this model type")
                return

            import shap
            n_features = X_val.shape[1]
            estimated_memory = (X_val.shape[0] * n_features * n_features * 8) / (1024 ** 3)
            
            if estimated_memory <= self.config.max_shap_interaction_memory_gb:
                explainer = shap.TreeExplainer(model)
                interaction_values = explainer.shap_interaction_values(X_val)
                
                if isinstance(interaction_values, list):
                    interaction_values = interaction_values[1]
                    
                results.shap_interaction_values = interaction_values
                structured_log(logger, logging.INFO, "Calculated SHAP interaction values",
                             interaction_shape=interaction_values.shape)
            else:
                structured_log(logger, logging.WARNING, 
                             "Skipping SHAP interactions due to memory constraints",
                             estimated_memory_gb=estimated_memory)
                
        except Exception as e:
            structured_log(logger, logging.ERROR, "Failed to calculate SHAP interactions",
                         error=str(e))

    def _calculate_model_score(self, y_true, y_pred, results: ModelTrainingResults = None) -> float:
        """
        Calculate model performance score based on the configured metric.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            results: ModelTrainingResults object containing eval_metric
        """
        from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, mean_absolute_error, roc_auc_score
        
        # Get the metric from results or default to accuracy
        metric = results.eval_metric if results else 'accuracy'
        
        if metric.lower() == 'accuracy':
            return accuracy_score(y_true, (y_pred >= 0.5).astype(int))
        elif metric.lower() in ['logloss', 'log_loss']:
            return -log_loss(y_true, y_pred)  # Negative so higher is better
        elif metric.lower() == 'auc':
            return roc_auc_score(y_true, y_pred)
        else:
            return accuracy_score(y_true, (y_pred >= 0.5).astype(int))

    def _generate_learning_curve_data(self, X_train, y_train, X_val, y_val, 
                                    fold: int, model_params: Dict,
                                    results: ModelTrainingResults) -> ModelTrainingResults:
        """Generate learning curve data with sample size increments."""
        try:
            n_samples = len(X_train)
            # Create geometrically spaced sample sizes
            train_sizes = np.geomspace(
                start=min(100, n_samples), 
                stop=n_samples,
                num=10,
                dtype=int
            )
            
            for size in train_sizes:
                indices = self._stratified_sample_indices(y_train, size)
                
                X_sample = X_train.iloc[indices]
                y_sample = y_train.iloc[indices]
                
                # Train model on sample
                model = self._initialize_model(results.model_name, model_params)
                model.fit(X_sample, y_sample)
                
                # Generate predictions
                train_pred = model.predict_proba(X_sample)[:, 1]
                val_pred = model.predict_proba(X_val)[:, 1]
                
                # Calculate scores
                train_score = self._calculate_model_score(y_sample, train_pred, results)
                val_score = self._calculate_model_score(y_val, val_pred, results)
                
                # Add to learning curve data
                results.learning_curve_data.add_iteration(
                    train_score=train_score,
                    val_score=val_score,
                    iteration=size
                )
            
            return results
        
        except Exception as e:
            structured_log(logger, logging.ERROR,
                         "Error in learning curve generation",
                         error=str(e))
            raise

    def _stratified_sample_indices(self, y, n_samples: int) -> np.ndarray:
        """Get indices for a stratified sample of the data.
        
        Args:
            y: Target values
            n_samples: Number of samples to select
            
        Returns:
            np.ndarray: Array of selected indices
        """
        from sklearn.model_selection import StratifiedShuffleSplit
        
        # If n_samples equals or exceeds the total samples, return all indices
        if n_samples >= len(y):
            return np.arange(len(y))
        
        # Convert n_samples to a float ratio if it's an integer
        train_size = n_samples / len(y)
        
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size)
        indices, _ = next(splitter.split(np.zeros(len(y)), y))
        return indices
