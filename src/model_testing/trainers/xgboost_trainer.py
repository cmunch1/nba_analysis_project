import logging
import numpy as np
import xgboost as xgb
from typing import Dict
from .base_trainer import BaseTrainer
from ..data_classes import ModelTrainingResults
from ...logging.logging_utils import structured_log
from ...error_handling.custom_exceptions import ModelTestingError
from ...config.config import AbstractConfig

logger = logging.getLogger(__name__)

class XGBoostTrainer(BaseTrainer):
    def __init__(self, config: AbstractConfig):
        """Initialize XGBoost trainer with configuration."""
        super().__init__(config)

    def train(self, X_train, y_train, X_val, y_val, fold: int, model_params: Dict, results: ModelTrainingResults) -> ModelTrainingResults:
        """Train an XGBoost model with the enhanced ModelTrainingResults."""
        structured_log(logger, logging.INFO, "Starting XGBoost training", 
                    input_shape=X_train.shape)
        
        try:
            # Create DMatrix objects
            dtrain = xgb.DMatrix(
                X_train, 
                label=y_train,
                feature_names=X_train.columns.tolist(),
                enable_categorical=self.config.enable_categorical
            )
            dval = xgb.DMatrix(
                X_val, 
                label=y_val,
                feature_names=X_val.columns.tolist(),
                enable_categorical=self.config.enable_categorical
            )

            # Initialize dict to store evaluation results
            evals_result = {}

            # Train model
            model = xgb.train(
                params=model_params,
                dtrain=dtrain,
                num_boost_round=self.config.XGB.num_boost_round,
                early_stopping_rounds=self.config.XGB.early_stopping_rounds,
                evals=[(dtrain, 'train'), (dval, 'eval')],
                verbose_eval=self.config.XGB.verbose_eval,
                evals_result=evals_result
            )

            # Store model and generate predictions
            results.model = model
            results.predictions = model.predict(dval)
            results.num_boost_round = self.config.XGB.num_boost_round
            results.early_stopping = self.config.XGB.early_stopping_rounds
            results.enable_categorical = self.config.enable_categorical
            results.categorical_features = self.config.categorical_features

            # Process learning curve data if requested
            if self.config.generate_learning_curve_data:
                self._process_learning_curve_data(evals_result, results)

            structured_log(logger, logging.INFO, "Generated predictions", 
                        predictions_shape=results.predictions.shape,
                        predictions_mean=float(np.mean(results.predictions)))

            # Calculate feature importance
            self._calculate_feature_importance(model, X_train, results)

            # Calculate SHAP values if configured
            if self.config.calculate_shap_values:
                self._calculate_shap_values(model, X_val, y_val, results)

            return results

        except Exception as e:
            raise ModelTestingError(
                "Error in XGBoost model training",
                error_message=str(e),
                dataframe_shape=X_train.shape
            )

    def _process_learning_curve_data(self, evals_result: Dict, results: ModelTrainingResults) -> None:
        """Process and store learning curve data."""
        eval_metric = list(evals_result['train'].keys())[0]
        results.learning_curve_data.metric_name = eval_metric

        for i, (train_score, val_score) in enumerate(zip(
            evals_result['train'][eval_metric], 
            evals_result['eval'][eval_metric]
        )):
            train_score_converted, val_score_converted = self._convert_metric_scores(
                train_score, 
                val_score,
                eval_metric
            )
            results.learning_curve_data.add_iteration(
                train_score=train_score_converted,
                val_score=val_score_converted,
                iteration=i
            )

    def _calculate_feature_importance(self, model, X_train, results: ModelTrainingResults) -> None:
        """Calculate and store feature importance scores."""
        try:
            importance_dict = model.get_score(importance_type='gain')
            results.feature_importance_scores = np.array([
                importance_dict.get(feature, 0) 
                for feature in X_train.columns
            ])
            results.feature_names = X_train.columns.tolist()
            
            structured_log(logger, logging.INFO, "Calculated feature importance",
                        num_features_with_importance=len(importance_dict))
            
        except Exception as e:
            structured_log(logger, logging.WARNING, "Failed to calculate feature importance",
                        error=str(e))
            results.feature_importance_scores = np.zeros(X_train.shape[1])

    def _calculate_shap_values(self, model, X_val, y_val, results: ModelTrainingResults) -> None:
        """Calculate and store SHAP values."""
        try:
            # Calculate SHAP values
            shap_values = model.predict(xgb.DMatrix(X_val), pred_contribs=True)
            
            structured_log(logger, logging.INFO, "SHAP value details",
                        shap_values_shape=shap_values.shape,
                        shap_values_nulls=np.sum(np.isnan(shap_values)),
                        X_val_shape=X_val.shape)
            
            # Remove the bias term (last column) from SHAP values
            if shap_values.shape[1] == X_val.shape[1] + 1:
                structured_log(logger, logging.INFO, 
                            "Removing bias term from SHAP values",
                            original_shape=shap_values.shape)
                shap_values = shap_values[:, :-1]
                structured_log(logger, logging.INFO, 
                            "SHAP values after bias removal",
                            new_shape=shap_values.shape)
            
            # Store feature data and SHAP values
            results.feature_data = X_val
            results.target_data = y_val
            results.shap_values = shap_values
            
            # Calculate interactions if configured
            if self.config.calculate_shap_interactions:
                self._calculate_shap_interactions(model, X_val, results)
                
        except Exception as e:
            structured_log(logger, logging.ERROR, "Failed to calculate SHAP values",
                        error=str(e),
                        error_type=type(e).__name__)
            raise

    def _calculate_shap_interactions(self, model, X_val, results: ModelTrainingResults) -> None:
        """Calculate and store SHAP interaction values."""
        n_features = X_val.shape[1]
        estimated_memory = (X_val.shape[0] * n_features * n_features * 8) / (1024 ** 3)
        
        if estimated_memory <= self.config.max_shap_interaction_memory_gb:
            interaction_values = model.predict(xgb.DMatrix(X_val), pred_interactions=True)
            # Remove bias term from interaction values if present
            if interaction_values.shape[1] == X_val.shape[1] + 1:
                interaction_values = interaction_values[:, :-1, :-1]
            results.shap_interaction_values = interaction_values
            structured_log(logger, logging.INFO, "Calculated SHAP interaction values",
                        interaction_shape=interaction_values.shape)

    def _convert_metric_scores(self, train_score: float, val_score: float, metric_name: str) -> tuple[float, float]:
        """Convert metric scores to a consistent format (higher is better)."""
        lower_is_better = ['logloss', 'binary_logloss', 'multi_logloss', 'rmse', 'mae']
        
        if any(metric in metric_name.lower() for metric in lower_is_better):
            return -train_score, -val_score
        
        return train_score, val_score 