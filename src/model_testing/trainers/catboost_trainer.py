import logging
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from typing import Dict, Tuple
from catboost import Pool, CatBoost
from .base_trainer import BaseTrainer
from .trainer_utils import TrainerUtils
from src.common.data_classes import ModelTrainingResults
from src.common.config_management.base_config_manager import BaseConfigManager
from src.common.app_logging.base_app_logger import BaseAppLogger
from src.common.error_handling.base_error_handler import BaseErrorHandler

class CatBoostTrainer(BaseTrainer):
    def __init__(self, 
                 config: BaseConfigManager, 
                 app_logger: BaseAppLogger, 
                 error_handler: BaseErrorHandler):
        """Initialize CatBoost trainer with configuration and dependencies."""
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler
        self.utils = TrainerUtils(app_logger, error_handler)
        
        self.app_logger.structured_log(
            logging.INFO, 
            "CatBoostTrainer initialized successfully",
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
    def train(self, X_train, y_train, X_val, y_val, fold: int, model_params: Dict, results: ModelTrainingResults) -> ModelTrainingResults:
        """Train a CatBoost model with unified log-based metric tracking."""
        try:
            self.app_logger.structured_log(
                logging.INFO, 
                "Starting CatBoost training", 
                input_shape=X_train.shape
            )
            
            # Create CatBoost pools
            train_pool = Pool(
                data=X_train,
                label=y_train,
                feature_names=X_train.columns.tolist(),
                cat_features=self.config.categorical_features if hasattr(self.config, 'categorical_features') else None
            )
            val_pool = Pool(
                data=X_val,
                label=y_val,
                feature_names=X_val.columns.tolist(),
                cat_features=self.config.categorical_features if hasattr(self.config, 'categorical_features') else None
            )

            # Ensure metrics are properly set in parameters
            eval_metric = model_params.get('eval_metric', 'Logloss')
            if isinstance(eval_metric, str):
                model_params['eval_metric'] = eval_metric
            elif isinstance(eval_metric, list):
                model_params['eval_metric'] = eval_metric[0]  # Use first metric as primary
            
            # Remove 'metrics' key if it exists to avoid conflicts
            model_params.pop('metrics', None)
            primary_metric = model_params['eval_metric']
            
            # Configure logging directory and verbosity
            log_dir = Path(self.config.log_path) / f'catboost_info/fold_{fold}'
            if log_dir.exists():
                shutil.rmtree(log_dir)  # Clean up any existing logs
            log_dir.mkdir(parents=True, exist_ok=True)
            
            model_params.update({
                'train_dir': str(log_dir),
                'verbose': self.config.CatBoost.verbose_eval if hasattr(self.config, 'CatBoost') and hasattr(self.config.CatBoost, 'verbose_eval') else 100
            })

            # Initialize and train model
            model = CatBoost(params=model_params)
            model.fit(
                train_pool,
                eval_set=val_pool,
                early_stopping_rounds=self.config.CatBoost.early_stopping_rounds if hasattr(self.config, 'CatBoost') and hasattr(self.config.CatBoost, 'early_stopping_rounds') else 10,
                use_best_model=True
            )

            # Store model and update basic information
            results.model = model
            results.num_boost_round = model.tree_count_
            results.early_stopping = self.config.CatBoost.early_stopping_rounds if hasattr(self.config, 'CatBoost') and hasattr(self.config.CatBoost, 'early_stopping_rounds') else 10
            results.categorical_features = self.config.categorical_features if hasattr(self.config, 'categorical_features') else []
            
            # Generate predictions
            loss_function = model_params.get('loss_function', '').lower()
            if loss_function in ['logloss', 'crossentropy']:
                raw_predictions = model.predict(val_pool, prediction_type='Probability')
                results.predictions = raw_predictions if len(raw_predictions.shape) == 1 else raw_predictions[:, 1]
            else:
                results.predictions = model.predict(val_pool)

            # Process training logs
            self._process_training_logs(results, log_dir, primary_metric)
            
            self.app_logger.structured_log(
                logging.INFO, 
                "Generated predictions", 
                predictions_shape=results.predictions.shape,
                predictions_mean=float(np.mean(results.predictions)),
                loss_function=loss_function
            )

            # Calculate feature importance
            self._calculate_feature_importance(model, X_train, results)

            # Calculate SHAP values if configured
            if hasattr(self.config, 'calculate_shap_values') and self.config.calculate_shap_values:
                self._calculate_shap_values(model, val_pool, y_val, results)

            return results

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'model_testing',
                "Error in CatBoost model training",
                error_message=str(e),
                input_shape=X_train.shape
            )

    def _process_training_logs(self, results: ModelTrainingResults, log_dir: Path, metric_name: str) -> None:
        """Process CatBoost training logs and update results."""
        try:
            # Read training metrics
            learn_path = log_dir / 'learn_error.tsv'
            test_path = log_dir / 'test_error.tsv'
            
            if learn_path.exists() and test_path.exists():
                learn_df = pd.read_csv(learn_path, sep='\t')
                test_df = pd.read_csv(test_path, sep='\t')
                
                # Find the metric column
                metric_cols = [col for col in learn_df.columns if col.lower() == metric_name.lower()]
                if metric_cols:
                    metric_col = metric_cols[0]
                    
                    # Update learning curve data
                    results.learning_curve_data.iterations = learn_df['iter'].tolist()
                    results.learning_curve_data.train_scores = learn_df[metric_col].tolist()
                    results.learning_curve_data.val_scores = test_df[metric_col].tolist()
                    results.learning_curve_data.metric_name = metric_name
                    
                    self.app_logger.structured_log(
                        logging.INFO, 
                        "Processed training logs",
                        iterations=len(results.learning_curve_data.iterations),
                        metric_name=metric_name
                    )
                else:
                    self.app_logger.structured_log(
                        logging.WARNING, 
                        f"Metric {metric_name} not found in logs",
                        available_metrics=list(learn_df.columns)
                    )
            else:
                self.app_logger.structured_log(
                    logging.WARNING, 
                    "Training logs not found",
                    learn_path_exists=learn_path.exists(),
                    test_path_exists=test_path.exists()
                )
            
        except Exception as e:
            self.app_logger.structured_log(
                logging.ERROR, 
                "Error processing training logs",
                error=str(e),
                metric_name=metric_name
            )

    def _calculate_feature_importance(self, model, X_train, results: ModelTrainingResults) -> None:
        """Calculate and store feature importance scores."""
        try:
            importance_scores = model.get_feature_importance()
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

    def _calculate_shap_values(self, model, val_pool: Pool, y_val, results: ModelTrainingResults) -> None:
        """Calculate and store SHAP values."""
        try:
            # Calculate SHAP values
            shap_values = model.get_feature_importance(
                val_pool,
                type='ShapValues'
            )
            
            # Process and store SHAP values
            if shap_values.shape[1] == val_pool.get_features().shape[1] + 1:
                shap_values = shap_values[:, :-1]  # Remove bias term
                
            results.feature_data = val_pool.get_features()
            results.target_data = y_val
            results.shap_values = shap_values
            
            self.app_logger.structured_log(
                logging.INFO, 
                "Calculated SHAP values",
                shap_values_shape=shap_values.shape
            )
            
            # Calculate interactions if configured
            if hasattr(self.config, 'calculate_shap_interactions') and self.config.calculate_shap_interactions:
                self._calculate_shap_interactions(model, val_pool, results)
            
        except Exception as e:
            self.app_logger.structured_log(
                logging.ERROR, 
                "Failed to calculate SHAP values",
                error=str(e),
                error_type=type(e).__name__
            )

    def _calculate_shap_interactions(self, model, val_pool: Pool, results: ModelTrainingResults) -> None:
        """Calculate and store SHAP interaction values."""
        try:
            n_features = val_pool.get_features().shape[1]
            estimated_memory = (val_pool.get_features().shape[0] * n_features * n_features * 8) / (1024 ** 3)
            
            if not hasattr(self.config, 'max_shap_interaction_memory_gb') or estimated_memory <= self.config.max_shap_interaction_memory_gb:
                interaction_values = model.get_feature_importance(
                    val_pool,
                    type='ShapInteractionValues'
                )
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