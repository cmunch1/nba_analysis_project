import logging
import numpy as np
import lightgbm as lgb
from typing import Dict
from .base_trainer import BaseTrainer
from .trainer_utils import TrainerUtils
from ...common.data_classes import ModelTrainingResults
from ...common.config_management.base_config_manager import BaseConfigManager
from ...common.app_logging.base_app_logger import BaseAppLogger
from ...common.error_handling.base_error_handler import BaseErrorHandler

class LightGBMTrainer(BaseTrainer):
    def __init__(self, config: BaseConfigManager, app_logger: BaseAppLogger, error_handler: BaseErrorHandler):
        """Initialize LightGBM trainer with configuration and dependencies."""
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler
        self.utils = TrainerUtils()
        
        self.app_logger.structured_log(
            logging.INFO, 
            "LightGBMTrainer initialized successfully",
            trainer_type=type(self).__name__
        )

    @property
    def log_performance(self):
        """Get the performance logging decorator from app_logger."""
        return self.app_logger.log_performance

    @log_performance
    def train(self, X_train, y_train, X_val, y_val, fold: int, model_params: Dict, results: ModelTrainingResults) -> ModelTrainingResults:
        """Train a LightGBM model with the enhanced ModelTrainingResults."""
        try:
            self.app_logger.structured_log(
                logging.INFO, 
                "Starting LightGBM training", 
                input_shape=X_train.shape
            )
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(
                X_train, 
                label=y_train,
                feature_name=X_train.columns.tolist(),
                categorical_feature=self.config.categorical_features
            )
            val_data = lgb.Dataset(
                X_val, 
                label=y_val,
                feature_name=X_val.columns.tolist(),
                categorical_feature=self.config.categorical_features,
                reference=train_data
            )

            # Initialize dict to store evaluation results
            evals_result = {}

            # Train model
            model = lgb.train(
                params=model_params,
                train_set=train_data,
                num_boost_round=self.config.LightGBM.num_boost_round,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'eval'],
                early_stopping_rounds=self.config.LightGBM.early_stopping_rounds,
                verbose_eval=self.config.LightGBM.verbose_eval,
                evals_result=evals_result
            )

            # Store model and generate predictions
            results.model = model
            results.predictions = model.predict(X_val)
            results.num_boost_round = self.config.LightGBM.num_boost_round
            results.early_stopping = self.config.LightGBM.early_stopping_rounds
            results.categorical_features = self.config.categorical_features

            # Process learning curve data if requested
            if self.config.generate_learning_curve_data:
                self.utils._process_learning_curve_data(evals_result, results)

            self.app_logger.structured_log(
                logging.INFO, 
                "Generated predictions", 
                predictions_shape=results.predictions.shape,
                predictions_mean=float(np.mean(results.predictions))
            )

            # Calculate feature importance
            self._calculate_feature_importance(model, X_train, results)

            # Calculate SHAP values if configured
            if self.config.calculate_shap_values:
                self._calculate_shap_values(model, X_val, y_val, results)

            return results

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'model_testing',
                "Error in LightGBM training",
                original_error=str(e),
                input_shape=X_train.shape
            )

    def _calculate_feature_importance(self, model, X_train, results: ModelTrainingResults) -> None:
        """Calculate and store feature importance scores."""
        try:
            # LightGBM provides feature importance directly as an array
            importance_scores = model.feature_importance(importance_type='gain')
            results.feature_importance_scores = importance_scores
            results.feature_names = X_train.columns.tolist()
            
            self.app_logger.structured_log(
                logging.INFO, 
                "Calculated feature importance",
                num_features_with_importance=np.sum(importance_scores > 0)
            )
            
        except Exception as e:
            self.app_logger.structured_log(
                logging.WARNING, 
                "Failed to calculate feature importance",
                error=str(e)
            )
            results.feature_importance_scores = np.zeros(X_train.shape[1])

    def _calculate_shap_values(self, model, X_val, y_val, results: ModelTrainingResults) -> None:
        """Calculate and store SHAP values."""
        try:
            # Calculate SHAP values
            shap_values = model.predict(X_val, pred_contrib=True)
            
            self.app_logger.structured_log(
                logging.INFO, 
                "SHAP value details",
                shap_values_shape=shap_values.shape,
                shap_values_nulls=np.sum(np.isnan(shap_values)),
                X_val_shape=X_val.shape
            )
            
            # Remove the bias term (last column) from SHAP values
            if shap_values.shape[1] == X_val.shape[1] + 1:
                self.app_logger.structured_log(
                    logging.INFO, 
                    "Removing bias term from SHAP values",
                    original_shape=shap_values.shape
                )
                shap_values = shap_values[:, :-1]
                self.app_logger.structured_log(
                    logging.INFO, 
                    "SHAP values after bias removal",
                    new_shape=shap_values.shape
                )
            
            # Store feature data and SHAP values
            results.feature_data = X_val
            results.target_data = y_val
            results.shap_values = shap_values
            
            # Calculate interactions if configured
            if self.config.calculate_shap_interactions:
                self._calculate_shap_interactions(model, X_val, results)
                
        except Exception as e:
            self.app_logger.structured_log(
                logging.ERROR, 
                "Failed to calculate SHAP values",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def _calculate_shap_interactions(self, model, X_val, results: ModelTrainingResults) -> None:
        """Calculate and store SHAP interaction values."""
        n_features = X_val.shape[1]
        estimated_memory = (X_val.shape[0] * n_features * n_features * 8) / (1024 ** 3)
        
        if estimated_memory <= self.config.max_shap_interaction_memory_gb:
            interaction_values = model.predict(X_val, pred_interactions=True)
            # Remove bias term from interaction values if present
            if interaction_values.shape[1] == X_val.shape[1] + 1:
                interaction_values = interaction_values[:, :-1, :-1]
            results.shap_interaction_values = interaction_values
            self.app_logger.structured_log(
                logging.INFO, 
                "Calculated SHAP interaction values",
                interaction_shape=interaction_values.shape
            )