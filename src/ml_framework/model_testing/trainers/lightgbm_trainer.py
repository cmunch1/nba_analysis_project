import logging
import numpy as np
import lightgbm as lgb
from typing import Dict, Tuple, Optional
from .base_trainer import BaseTrainer
from .trainer_utils import TrainerUtils
from ml_framework.framework.data_classes import ModelTrainingResults
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler
from ml_framework.preprocessing.base_preprocessor import BasePreprocessor

class LightGBMTrainer(BaseTrainer):
    def __init__(self, config: BaseConfigManager, app_logger: BaseAppLogger, error_handler: BaseErrorHandler):
        """Initialize LightGBM trainer with configuration and dependencies."""
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler
        self.utils = TrainerUtils(app_logger, error_handler)
        self._model_cfg = config.core.model_testing_config
        
        self.app_logger.structured_log(
            logging.INFO, 
            "LightGBMTrainer initialized successfully",
            trainer_type=type(self).__name__
        )

    @staticmethod
    def log_performance(func):
        """Decorator factory for performance logging"""
        def wrapper(*args, **kwargs):
            # Get the self instance from args since this is now a static method
            instance = args[0]
            return instance.app_logger.log_performance(func)(*args, **kwargs)
        return wrapper

    @log_performance
    def train(self, X_train, y_train, X_val, y_val, fold: int, model_params: Dict, results: ModelTrainingResults,
              preprocessor: Optional[BasePreprocessor] = None) -> ModelTrainingResults:
        """
        Train a LightGBM model with optional preprocessing.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            fold: Current fold number
            model_params: Model hyperparameters
            results: ModelTrainingResults object to store results
            preprocessor: Optional preprocessor for model-specific transforms

        Returns:
            Updated ModelTrainingResults object with training results
        """
        try:
            self.app_logger.structured_log(
                logging.INFO,
                "Starting LightGBM training",
                input_shape=X_train.shape,
                has_preprocessor=preprocessor is not None
            )

            # Apply preprocessing if provided
            if preprocessor:
                self.app_logger.structured_log(
                    logging.INFO,
                    "Applying preprocessing for LightGBM",
                    fold=fold
                )

                # Fit preprocessor on training data and transform
                X_train_transformed, prep_results = preprocessor.fit_transform(
                    X_train,
                    y_train,
                    model_name='LightGBM'
                )

                # Transform validation data (don't refit!)
                X_val_transformed = preprocessor.transform(X_val)

                # Store preprocessing artifact in results
                results.preprocessing_artifact = preprocessor.get_preprocessor_artifact()

                self.app_logger.structured_log(
                    logging.INFO,
                    "Preprocessing completed",
                    train_shape=X_train_transformed.shape,
                    val_shape=X_val_transformed.shape
                )
            else:
                X_train_transformed = X_train
                X_val_transformed = X_val

            # Create LightGBM datasets
            train_data = lgb.Dataset(
                X_train_transformed,
                label=y_train,
                feature_name=X_train_transformed.columns.tolist(),
                categorical_feature=self._model_cfg.categorical_features if hasattr(self._model_cfg, 'categorical_features') else None
            )
            val_data = lgb.Dataset(
                X_val_transformed,
                label=y_val,
                feature_name=X_val_transformed.columns.tolist(),
                categorical_feature=self._model_cfg.categorical_features if hasattr(self._model_cfg, 'categorical_features') else None,
                reference=train_data
            )

            # Initialize dict to store evaluation results
            evals_result = {}

            # Prepare callbacks for early stopping and verbosity
            callbacks = []

            # Add early stopping callback if configured
            early_stopping_rounds = self.config.LightGBM.early_stopping_rounds if hasattr(self.config, 'LightGBM') and hasattr(self.config.LightGBM, 'early_stopping_rounds') else 10
            if early_stopping_rounds:
                callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))

            # Add verbose evaluation callback if configured
            verbose_eval = self.config.LightGBM.verbose_eval if hasattr(self.config, 'LightGBM') and hasattr(self.config.LightGBM, 'verbose_eval') else 100
            if verbose_eval:
                callbacks.append(lgb.log_evaluation(period=verbose_eval))

            # Add callback to record evaluation results
            callbacks.append(lgb.record_evaluation(evals_result))

            # Train model
            model = lgb.train(
                params=model_params,
                train_set=train_data,
                num_boost_round=self.config.LightGBM.num_boost_round if hasattr(self.config, 'LightGBM') and hasattr(self.config.LightGBM, 'num_boost_round') else 100,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'eval'],
                callbacks=callbacks
            )

            # Store model and generate predictions
            results.model = model
            results.predictions = model.predict(X_val_transformed)
            results.num_boost_round = self.config.LightGBM.num_boost_round if hasattr(self.config, 'LightGBM') and hasattr(self.config.LightGBM, 'num_boost_round') else 100
            results.early_stopping = self.config.LightGBM.early_stopping_rounds if hasattr(self.config, 'LightGBM') and hasattr(self.config.LightGBM, 'early_stopping_rounds') else 10
            results.categorical_features = self._model_cfg.categorical_features if hasattr(self._model_cfg, 'categorical_features') else []

            # Process learning curve data if requested
            if hasattr(self.config, 'generate_learning_curve_data') and self.config.generate_learning_curve_data:
                self._process_learning_curve_data(evals_result, results)

            self.app_logger.structured_log(
                logging.INFO,
                "Generated predictions",
                predictions_shape=results.predictions.shape,
                predictions_mean=float(np.mean(results.predictions))
            )

            # Calculate feature importance (use transformed data if preprocessing was applied)
            X_for_importance = X_train_transformed if preprocessor else X_train
            self._calculate_feature_importance(model, X_for_importance, results)

            # Calculate SHAP values if configured (use transformed data if preprocessing was applied)
            if hasattr(self._model_cfg, 'calculate_shap_values') and self._model_cfg.calculate_shap_values:
                X_for_shap = X_val_transformed if preprocessor else X_val
                self._calculate_shap_values(model, X_for_shap, y_val, results)

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
            if hasattr(self._model_cfg, 'calculate_shap_interactions') and self._model_cfg.calculate_shap_interactions:
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

        if not hasattr(self._model_cfg, 'max_shap_interaction_memory_gb') or estimated_memory <= self._model_cfg.max_shap_interaction_memory_gb:
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

    def _convert_metric_scores(self, train_score: float, val_score: float, metric_name: str) -> Tuple[float, float]:
        """Convert metric scores to a consistent format (higher is better)."""
        return self.utils._convert_metric_scores(train_score, val_score, metric_name)

    def _process_learning_curve_data(self, evals_result: Dict, results: ModelTrainingResults) -> None:
        """Process and store learning curve data."""
        return self.utils._process_learning_curve_data(evals_result, results)