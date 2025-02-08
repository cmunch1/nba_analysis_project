import logging
import numpy as np
import pandas as pd
from catboost import Pool, CatBoost
from typing import Dict, List
import shutil
from pathlib import Path
from .base_trainer import BaseTrainer
from ..data_classes import ModelTrainingResults
from ...logging.logging_utils import structured_log
from ...error_handling.custom_exceptions import ModelTestingError
from ...config.config import AbstractConfig


logger = logging.getLogger(__name__)

class CatBoostTrainer(BaseTrainer):
    def __init__(self, config: AbstractConfig):
        """Initialize CatBoost trainer with configuration."""
        super().__init__(config)

    def train(self, X_train, y_train, X_val, y_val, fold: int, model_params: Dict, results: ModelTrainingResults) -> ModelTrainingResults:
        """Train a CatBoost model with unified log-based metric tracking."""
        structured_log(logger, logging.INFO, "Starting CatBoost training", 
                    input_shape=X_train.shape)
        
        try:
            # Create CatBoost pools
            train_pool = Pool(
                data=X_train,
                label=y_train,
                feature_names=X_train.columns.tolist(),
                cat_features=self.config.categorical_features
            )
            val_pool = Pool(
                data=X_val,
                label=y_val,
                feature_names=X_val.columns.tolist(),
                cat_features=self.config.categorical_features
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

            })

            # Initialize and train model
            model = CatBoost(params=model_params)
            model.fit(
                train_pool,
                eval_set=val_pool,
                early_stopping_rounds=self.config.CatBoost.early_stopping_rounds,
                verbose_eval=self.config.CatBoost.verbose_eval
            )

            # Store model and update basic information
            results.model = model
            results.num_boost_round = model.tree_count_
            results.early_stopping = self.config.CatBoost.early_stopping_rounds
            results.categorical_features = self.config.categorical_features
            
            # Generate predictions
            loss_function = model_params.get('loss_function', '').lower()
            if loss_function in ['logloss', 'crossentropy']:
                raw_predictions = model.predict(val_pool, prediction_type='Probability')
                results.predictions = raw_predictions if len(raw_predictions.shape) == 1 else raw_predictions[:, 1]
            else:
                results.predictions = model.predict(val_pool)

            # Parse training logs and update results
            results = self._process_training_logs(results, fold, primary_metric)
            
            structured_log(logger, logging.INFO, "Generated predictions", 
                        predictions_shape=results.predictions.shape,
                        predictions_mean=float(np.mean(results.predictions)),
                        loss_function=loss_function)

            # Calculate feature importance
            self._calculate_feature_importance(model, X_train, results)

            # Calculate SHAP values if configured
            if self.config.calculate_shap_values:
                self._calculate_shap_values(model, X_val, y_val, results)

            return results

        except Exception as e:
            raise ModelTestingError(
                "Error in CatBoost model training",
                error_message=str(e),
                dataframe_shape=X_train.shape
            )

    def _process_training_logs(self, results: ModelTrainingResults, fold: int, metric_name: str) -> ModelTrainingResults:
        """Process CatBoost training logs and update results."""
        try:
            log_dir = Path(self.config.log_path) / f'catboost_info/fold_{fold}'
            

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
                    
                    structured_log(logger, logging.INFO, "Processed training logs",
                                iterations=len(results.learning_curve_data.iterations),
                                metric_name=metric_name)
                else:
                    structured_log(logger, logging.WARNING, 
                                f"Metric {metric_name} not found in logs",
                                available_metrics=list(learn_df.columns))
            else:
                structured_log(logger, logging.WARNING, "Training logs not found",
                            learn_path_exists=learn_path.exists(),
                            test_path_exists=test_path.exists())
                            
            return results
            
        except Exception as e:
            structured_log(logger, logging.ERROR, "Error processing training logs",
                        error=str(e),
                        fold=fold,
                        metric_name=metric_name)
            return results

    def _calculate_feature_importance(self, model, X_train, results: ModelTrainingResults) -> None:
        """Calculate and store feature importance scores."""
        try:
            importance_scores = model.get_feature_importance()
            results.feature_importance_scores = importance_scores
            results.feature_names = X_train.columns.tolist()
            
            structured_log(logger, logging.INFO, "Calculated feature importance",
                        num_features_with_importance=np.sum(importance_scores > 0))
            
        except Exception as e:
            structured_log(logger, logging.WARNING, "Failed to calculate feature importance",
                        error=str(e))
            results.feature_importance_scores = np.zeros(X_train.shape[1])

    def _calculate_shap_values(self, model, X_val, y_val, results: ModelTrainingResults) -> None:
        """Calculate and store SHAP values."""
        try:
            # Create validation pool with categorical features properly handled
            if isinstance(X_val, pd.DataFrame):
                for col in self.config.categorical_features:
                    if col in X_val.columns:
                        X_val[col] = X_val[col].astype('category')
            
            val_pool = Pool(
                data=X_val,
                label=y_val,
                cat_features=self.config.categorical_features
            )
            
            # Calculate SHAP values
            shap_values = model.get_feature_importance(
                val_pool,
                type='ShapValues'
            )
            
            # Process and store SHAP values
            if shap_values.shape[1] == X_val.shape[1] + 1:
                shap_values = shap_values[:, :-1]  # Remove bias term
                
            results.feature_data = X_val
            results.target_data = y_val
            results.shap_values = shap_values
            
            structured_log(logger, logging.INFO, "Calculated SHAP values",
                        shap_values_shape=shap_values.shape)
            
        except Exception as e:
            structured_log(logger, logging.ERROR, "Failed to calculate SHAP values",
                        error=str(e),
                        error_type=type(e).__name__)

    def _calculate_shap_interactions(self, model, val_pool: Pool, results: ModelTrainingResults) -> None:
        """Calculate and store SHAP interaction values."""
        n_features = val_pool.get_features().shape[1]
        estimated_memory = (val_pool.get_features().shape[0] * n_features * n_features * 8) / (1024 ** 3)
        
        if estimated_memory <= self.config.max_shap_interaction_memory_gb:
            interaction_values = model.get_feature_importance(
                val_pool,
                type='ShapInteractionValues'
            )
            results.shap_interaction_values = interaction_values
            structured_log(logger, logging.INFO, "Calculated SHAP interaction values",
                        interaction_shape=interaction_values.shape)