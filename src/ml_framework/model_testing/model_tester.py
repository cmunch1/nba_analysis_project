"""
Model testing implementation with support for nested configurations,
proper dependency injection, and enhanced error handling.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Union, Any, List, Optional
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler
from ml_framework.framework.data_classes import (
    ModelTrainingResults,
    ClassificationMetrics,
    PreprocessingResults
)
from ml_framework.preprocessing.base_preprocessor import BasePreprocessor
from ml_framework.visualization.orchestration.base_chart_orchestrator import BaseChartOrchestrator

from .base_model_testing import BaseModelTester
from .hyperparams_managers.base_hyperparams_manager import BaseHyperparamsManager
from .trainers.base_trainer import BaseTrainer

class ModelTester(BaseModelTester):
    def __init__(self, 
                 config: BaseConfigManager,
                 hyperparameter_manager: BaseHyperparamsManager,
                 trainers: Dict[str, BaseTrainer],
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler,
                 preprocessor: BasePreprocessor,
                 chart_orchestrator: BaseChartOrchestrator):
        """
        Initialize the ModelTester with injected dependencies.

        Args:
            config: Configuration manager
            hyperparameter_manager: Manager for model hyperparameters
            trainers: Dictionary mapping model names to their trainers
            app_logger: Application logger
            error_handler: Error handling utility
        """
        self.config = config
        self._model_cfg = config.core.model_testing_config
        self.hyperparameter_manager = hyperparameter_manager
        self.preprocessor = preprocessor
        self.trainers = trainers
        self.app_logger = app_logger
        self.error_handler = error_handler

        self.app_logger.structured_log(logging.INFO, "ModelTester initialized",
                                     config_type=type(config).__name__,
                                     available_trainers=list(trainers.keys()))

    @staticmethod
    def log_performance(func):
        """Decorator factory for performance logging"""
        def wrapper(*args, **kwargs):
            # Get the self instance from args since this is now a static method
            instance = args[0]
            return instance.app_logger.log_performance(func)(*args, **kwargs)
        return wrapper

    @log_performance
    def get_model_params(self, model_name: str) -> Dict:
        """Get the current best hyperparameters for a model."""
        try:
            return self.hyperparameter_manager.get_current_params(model_name)
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'model_testing',
                f"Error getting hyperparameters for {model_name}",
                original_error=str(e)
            )

    @log_performance
    def prepare_data(self, df: pd.DataFrame, model_name: str = None, 
                    is_training: bool = True, 
                    preprocessing_results: PreprocessingResults = None) -> Tuple[pd.DataFrame, pd.Series, PreprocessingResults, pd.Series]:
        """
        Prepare data for model training or validation with preprocessing.
        
        Args:
            df: Input dataframe
            model_name: Name of the model
            is_training: Whether this is training data
            preprocessing_results: Optional existing preprocessing results
            
        Returns:
            Tuple containing:
            - Feature dataframe
            - Target series
            - Preprocessing results
            - Primary IDs series
        """
        self.app_logger.structured_log(logging.INFO, "Starting data preparation",
                                     input_shape=df.shape,
                                     is_training=is_training)
        
        primary_ids = None

        try:
            # Get sort configuration from model-specific config if available
            sort_columns = self.get_model_config_value(model_name, 'sort_columns',
                                                      self._model_cfg.sort_columns)
            sort_order = self.get_model_config_value(model_name, 'sort_order',
                                                    self._model_cfg.sort_order)

            # Sort chronologically for time series data
            df = df.sort_values(by=sort_columns, ascending=sort_order)

            # Extract target and features
            target_column = self.config.home_team_prefix + self.config.target_column
            y = df[target_column]
            X = df.drop(columns=[target_column])

            # Handle primary ID column
            if hasattr(self._model_cfg, 'primary_id_column') and self._model_cfg.primary_id_column:
                primary_ids = X[self._model_cfg.primary_id_column]
                X = X.drop(columns=[self._model_cfg.primary_id_column])

            # Drop non-useful columns
            if hasattr(self._model_cfg, 'non_useful_columns') and self._model_cfg.non_useful_columns:
                X = X.drop(columns=self._model_cfg.non_useful_columns)

            # Get preprocessing configuration
            perform_preprocessing = self.get_model_config_value(
                model_name, 'perform_preprocessing',
                self._model_cfg.perform_preprocessing
            )

            if perform_preprocessing:
                # Store original column names
                original_columns = X.columns.tolist()

                # Get categorical features from model config
                categorical_features = self.get_model_config_value(
                    model_name, 'categorical_features',
                    self._model_cfg.categorical_features
                )

                # Convert categorical features
                for col in categorical_features:
                    if col in X.columns:
                        X[col] = X[col].astype('category')

                # Apply preprocessing
                if is_training:
                    X, preprocessing_results = self.preprocessor.fit_transform(
                        X,
                        y=y,
                        model_name=model_name,
                        preprocessing_results=preprocessing_results
                    )
                else:
                    X = self.preprocessor.transform(X)

                # Restore column names if possible
                if isinstance(X, pd.DataFrame):
                    if len(X.columns) == len(original_columns):
                        X.columns = original_columns
                else:
                    X = pd.DataFrame(X, columns=original_columns)

                # Log preprocessing results
                if is_training:
                    steps_summary = [
                        {
                            'name': step.name,
                            'type': step.type,
                            'n_columns': len(step.columns)
                        }
                        for step in preprocessing_results.steps
                    ]
                    self.app_logger.structured_log(logging.INFO, "Preprocessing results",
                                                steps_overview=steps_summary)

            # Optimize memory usage
            X = self._reduce_memory_footprint(X)
            
            self.app_logger.structured_log(logging.INFO, "Data preparation completed",
                                         output_shape=X.shape)

            return X, y, preprocessing_results, primary_ids

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'model_testing',
                "Error in data preparation",
                original_error=str(e),
                dataframe_shape=df.shape
            )

    @log_performance
    def perform_oof_cross_validation(self, X: pd.DataFrame, y: pd.Series, 
                                   model_name: str, model_params: Dict,
                                   full_results: ModelTrainingResults) -> ModelTrainingResults:
        """Perform Out-of-Fold cross-validation with model-specific settings."""
        self.app_logger.structured_log(logging.INFO, f"{model_name} - Starting OOF cross-validation",
                                     input_shape=X.shape)
        try:
            # Initialize results arrays
            full_results.predictions = np.full(len(y), np.nan)
            full_results.target_data = y.copy()

            # Get model-specific configuration
            model_config = self.get_model_config(model_name)
            
            # Initialize SHAP arrays if needed
            if self.get_model_config_value(model_name, 'calculate_shap_values',
                                          self._model_cfg.calculate_shap_values):
                full_results.shap_values = np.full((len(y), X.shape[1]), np.nan)
                if self.get_model_config_value(model_name, 'calculate_shap_interactions',
                                              self._model_cfg.calculate_shap_interactions):
                    full_results.shap_interaction_values = np.full((len(y), X.shape[1], X.shape[1]), np.nan)

            # Setup cross-validation
            n_splits = self.get_model_config_value(model_name, 'n_splits', self._model_cfg.n_splits)
            cv_type = self.get_model_config_value(model_name, 'cross_validation_type',
                                                 self._model_cfg.cross_validation_type)
            
            if cv_type == "StratifiedKFold":
                kf = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                                   random_state=self.config.random_state)
            elif cv_type == "TimeSeriesSplit":
                kf = TimeSeriesSplit(n_splits=n_splits)
            else:
                raise ValueError(f"Unsupported CV type: {cv_type}")

            # Initialize tracking variables
            feature_importance_accumulator = np.zeros(X.shape[1])
            n_folds_with_importance = 0
            n_folds_with_curves = 0
            processed_samples = np.zeros(len(y), dtype=bool)

            # Perform cross-validation
            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                fold_results = ModelTrainingResults(X_train.shape)
                oof_results = self._train_model(
                    X_train, y_train,
                    X_val, y_val,
                    fold, model_name, model_params,
                    fold_results
                )

                # Update results from first fold
                if fold == 1:
                    self._update_first_fold_results(full_results, oof_results)

                # Store predictions and update processed samples
                full_results.predictions[val_idx] = oof_results.predictions
                processed_samples[val_idx] = True

                # Update SHAP values if available
                self._update_shap_values(full_results, oof_results, val_idx)

                # Update feature importance
                self._update_feature_importance(full_results, oof_results,
                                             feature_importance_accumulator,
                                             n_folds_with_importance)

                # Update learning curves
                self._update_learning_curves(full_results, oof_results, n_folds_with_curves)

            # Handle unprocessed samples for TimeSeriesSplit
            if not np.all(processed_samples):
                self._handle_unprocessed_samples(full_results, processed_samples, X, y)

            # Finalize results
            self._finalize_results(full_results, n_folds_with_importance, n_folds_with_curves)

            return full_results

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'model_testing',
                "Error in OOF cross-validation",
                original_error=str(e),
                dataframe_shape=X.shape,
                model_name=model_name,
                model_params=model_params,
            )
        
    @log_performance    
    def perform_validation_set_testing(self, X: pd.DataFrame, y: pd.Series, 
                                    X_val: pd.DataFrame, y_val: pd.Series, 
                                    model_name: str, model_params: Dict, results: ModelTrainingResults) -> ModelTrainingResults:
        """
        Perform model training and validation on a separate validation set.
        """
        try:
            fold = 1 # needs a value for consistency with OOF cross-validation
            
            results = self._train_model(
                X, y,
                X_val, y_val, fold,
                model_name, model_params,
                results
            )

            
            return results
            
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'model_testing',
                "Error in validation set testing",
                original_error=str(e),
                dataframe_shape=X.shape,
                model_name=model_name,
                model_params=model_params,
            )
        
    @log_performance
    def calculate_classification_evaluation_metrics(self, y_true, y_prob, metrics: ClassificationMetrics) -> ClassificationMetrics:
        """Calculate classification metrics using probability scores and optimal threshold."""
        
        self.app_logger.structured_log(logging.INFO, "Calculating classification metrics")
        
        try:
            # Input validation
            if y_true is None or y_prob is None:
                raise ValueError("y_true and y_prob must not be None")
            
            if not isinstance(y_true, (np.ndarray, pd.Series)) or not isinstance(y_prob, (np.ndarray, pd.Series)):
                raise ValueError("y_true and y_prob must be numpy arrays or pandas Series")

            if np.any(np.isnan(y_prob)):
                self.app_logger.structured_log(logging.WARNING, 
                            "Unexpected NaN values found in predictions after OOF filtering",
                            nan_count=np.sum(np.isnan(y_prob)))
                # Filter out NaN values
                mask = ~np.isnan(y_prob)
                y_true = y_true[mask]
                y_prob = y_prob[mask]
                
                if len(y_true) == 0:
                    raise ValueError("No valid predictions after filtering NaN values")
            
            # Find optimal threshold using ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            # Convert probabilities to predictions using optimal threshold
            y_pred = (y_prob >= optimal_threshold).astype(int)
            
            # Calculate metrics
            metrics.accuracy = accuracy_score(y_true, y_pred)
            metrics.precision = precision_score(y_true, y_pred)
            metrics.recall = recall_score(y_true, y_pred)
            metrics.f1 = f1_score(y_true, y_pred)
            metrics.auc = roc_auc_score(y_true, y_prob)
            metrics.optimal_threshold = float(optimal_threshold)
            
            # Add information about samples
            metrics.valid_samples = len(y_true)
            metrics.total_samples = len(y_true)  # These are now equal since filtering happened earlier
            metrics.nan_percentage = 0  # No NaNs should be present
            
            self.app_logger.structured_log(logging.INFO, "Classification metrics calculated",
                        optimal_threshold=float(optimal_threshold),
                        accuracy=float(metrics.accuracy),
                        auc=float(metrics.auc),
                        n_samples=len(y_true))
            
            return metrics
            
        except Exception as e:
            self.app_logger.structured_log(logging.ERROR, "Error calculating classification metrics",
                        error=str(e),
                        y_true_type=type(y_true).__name__ if y_true is not None else None,
                        y_prob_type=type(y_prob).__name__ if y_prob is not None else None,
                        y_true_shape=y_true.shape if hasattr(y_true, 'shape') else None,
                        y_prob_shape=y_prob.shape if hasattr(y_prob, 'shape') else None)
            raise self.error_handler.create_error_handler(
                'model_testing',
                "Error calculating classification metrics",
                original_error=str(e),
                n_samples=len(y_true) if y_true is not None else None,
                n_predictions=len(y_prob) if y_prob is not None else None)
    


    def get_model_config(self, model_name: str) -> Any:
        """Get model-specific configuration."""
        try:
            config_name = model_name.replace('_', '')  # Remove underscores for config access
            return getattr(self._model_cfg.models, config_name, None)
        except AttributeError:
            self.app_logger.structured_log(
                logging.WARNING,
                f"No configuration found for model: {model_name}"
            )
            return None
    
    def get_model_config_value(self, model_name: str, key: str, default: Any) -> Any:
        """Get a configuration value with fallback to default."""

        model_key = None
        model_config = self.get_model_config(model_name)
        # check if the model.yaml file has the key if not use the default value from model_testing_config.yaml
        if model_config is not None and hasattr(model_config, key):
            model_key = getattr(model_config, key)
            self.app_logger.structured_log(logging.INFO, "Model specific config value found",
                                           model_name=model_name,
                                           key=key,
                                           value=model_key)
        else:
            model_key = default       

        return model_key

    def _get_trainer(self, model_name: str) -> BaseTrainer:
        """Get the appropriate trainer for the model."""
        # Convert to uppercase for comparison
        model_name = model_name.upper()
        
        # Try exact match first
        if model_name in self.trainers:
            return self.trainers[model_name]
            
        # Try case-insensitive match
        for trainer_name in self.trainers:
            if trainer_name.upper() == model_name:
                return self.trainers[trainer_name]
                
        raise self.error_handler.create_error_handler(
            'model_testing',
            f"No trainer found for model: {model_name}",
            available_trainers=list(self.trainers.keys())
        )

    @log_performance
    def _reduce_memory_footprint(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataframe memory usage."""
        self.app_logger.structured_log(logging.INFO, "Starting memory footprint reduction",
                                     input_shape=df.shape, 
                                     input_memory=df.memory_usage().sum() / 1e6)
        try:
            # Optimize numeric columns
            for col in df.select_dtypes(include=['int', 'float']).columns:
                col_min, col_max = df[col].min(), df[col].max()

                # Integer optimization
                if df[col].dtype.kind in ['i', 'u']:
                    if col_min >= -128 and col_max <= 127:
                        df[col] = df[col].astype(np.int8)
                    elif col_min >= -32768 and col_max <= 32767:
                        df[col] = df[col].astype(np.int16)
                    elif col_min >= -2147483648 and col_max <= 2147483647:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
                # Float optimization
                else:
                    df[col] = df[col].astype(np.float32)

            self.app_logger.structured_log(logging.INFO, "Memory footprint reduction completed",
                                         output_shape=df.shape, 
                                         output_memory=df.memory_usage().sum() / 1e6)
            return df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'model_testing',
                "Error in memory footprint reduction",
                original_error=str(e),
                dataframe_shape=df.shape
            )

    @log_performance
    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series, fold: int,
                    model_name: str, model_params: Dict,
                    results: ModelTrainingResults) -> ModelTrainingResults:
        """Train a model on the given data."""
        self.app_logger.structured_log(logging.INFO, f"{model_name} - Starting model training",
                                     input_shape=X_train.shape)
        try:
            # Debug column alignment
            self.app_logger.structured_log(logging.DEBUG, "Column comparison",
                                         train_columns=X_train.columns.tolist(),
                                         val_columns=X_val.columns.tolist())

            # Verify column alignment
            missing_cols = set(X_train.columns) - set(X_val.columns)
            extra_cols = set(X_val.columns) - set(X_train.columns)
            
            if missing_cols or extra_cols:
                self.app_logger.structured_log(logging.WARNING, "Column mismatch detected",
                                             missing_columns=list(missing_cols),
                                             extra_columns=list(extra_cols))

            # Store basic information
            results.model_name = model_name
            results.model_params = model_params           
            results.feature_names = X_train.columns.tolist()
            results.update_feature_data(X_val, y_val)
            
            # Get appropriate trainer and train model
            trainer = self._get_trainer(model_name)
            results = trainer.train(X_train, y_train, X_val, y_val, fold, model_params, results)

            if results.predictions is None:
                raise self.error_handler.create_error_handler(
                    'model_testing',
                    "Model training completed but predictions are None"
                )
                                    
            self.app_logger.structured_log(logging.INFO, f"{model_name} - Model training completed",
                                         predictions_shape=results.predictions.shape)
            
            return results
        
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'model_testing',
                "Error in model training",
                original_error=str(e),
                model_name=model_name,
                dataframe_shape=X_train.shape
            )

    def _update_first_fold_results(self, full_results: ModelTrainingResults, 
                                 fold_results: ModelTrainingResults) -> None:
        """Update results with information from first fold."""
        full_results.num_boost_round = fold_results.num_boost_round
        full_results.early_stopping = fold_results.early_stopping
        full_results.enable_categorical = fold_results.enable_categorical
        full_results.categorical_features = fold_results.categorical_features
        full_results.model = fold_results.model
        full_results.feature_names = fold_results.feature_names

    def _update_shap_values(self, full_results: ModelTrainingResults,
                          fold_results: ModelTrainingResults,
                          val_idx: np.ndarray) -> None:
        """Update SHAP values from fold results."""
        if fold_results.shap_values is not None and hasattr(full_results, 'shap_values'):
            full_results.shap_values[val_idx] = fold_results.shap_values

            if (fold_results.shap_interaction_values is not None and 
                hasattr(full_results, 'shap_interaction_values')):
                full_results.shap_interaction_values[val_idx] = fold_results.shap_interaction_values

    def _update_feature_importance(self, full_results: ModelTrainingResults,
                                fold_results: ModelTrainingResults,
                                importance_accumulator: np.ndarray,
                                n_folds_with_importance: int) -> None:
        """Update feature importance scores."""
        if fold_results.feature_importance_scores is not None:
            importance_accumulator += fold_results.feature_importance_scores
            n_folds_with_importance += 1

    def _update_learning_curves(self, full_results: ModelTrainingResults,
                             fold_results: ModelTrainingResults,
                             n_folds_with_curves: int) -> None:
        """Update learning curve data from fold results."""
        if fold_results.learning_curve_data.train_scores:
            if len(full_results.learning_curve_data.train_scores) == 0:
                # Store first fold's data
                full_results.learning_curve_data.train_scores = np.array(
                    fold_results.learning_curve_data.train_scores
                )
                full_results.learning_curve_data.val_scores = np.array(
                    fold_results.learning_curve_data.val_scores
                )
                full_results.learning_curve_data.iterations = (
                    fold_results.learning_curve_data.iterations
                )
                full_results.learning_curve_data.metric_name = (
                    fold_results.learning_curve_data.metric_name
                )
                n_folds_with_curves = 1
            else:
                # Average with existing data
                min_length = min(
                    len(full_results.learning_curve_data.train_scores),
                    len(fold_results.learning_curve_data.train_scores)
                )
                
                full_results.learning_curve_data.train_scores = (
                    full_results.learning_curve_data.train_scores[:min_length] + 
                    np.array(fold_results.learning_curve_data.train_scores[:min_length])
                )
                full_results.learning_curve_data.val_scores = (
                    full_results.learning_curve_data.val_scores[:min_length] + 
                    np.array(fold_results.learning_curve_data.val_scores[:min_length])
                )
                full_results.learning_curve_data.iterations = (
                    full_results.learning_curve_data.iterations[:min_length]
                )
                n_folds_with_curves += 1

                self.app_logger.structured_log(logging.INFO, "Learning curve data combined",
                                            current_fold=n_folds_with_curves,
                                            iterations_used=min_length,
                                            original_iterations=len(
                                                fold_results.learning_curve_data.train_scores
                                            ))

    @log_performance
    def _handle_unprocessed_samples(self, full_results: ModelTrainingResults,
                                 processed_samples: np.ndarray,
                                 X: pd.DataFrame, y: pd.Series) -> None:
        """Handle samples not processed during cross-validation."""
        unprocessed_count = np.sum(~processed_samples)
        self.app_logger.structured_log(logging.WARNING,
                                     "Some samples were not processed in cross-validation",
                                     unprocessed_count=unprocessed_count,
                                     cv_type=self._model_cfg.cross_validation_type)
        
        # Keep only processed samples in the results
        processed_mask = processed_samples
        if full_results.feature_data is None:
            full_results.feature_data = X
            full_results.target_data = y
            
        full_results.feature_data = X[processed_mask]
        full_results.target_data = y[processed_mask]
        full_results.predictions = full_results.predictions[processed_mask]
        
        # Filter SHAP values
        if full_results.shap_values is not None:
            self.app_logger.structured_log(logging.INFO, 
                                         "Filtering SHAP values for processed samples")
            full_results.shap_values = full_results.shap_values[processed_mask]
            if full_results.shap_interaction_values is not None:
                full_results.shap_interaction_values = (
                    full_results.shap_interaction_values[processed_mask]
                )

    def _finalize_results(self, full_results: ModelTrainingResults,
                        n_folds_with_importance: int,
                        n_folds_with_curves: int) -> None:
        """Finalize aggregated results from all folds."""
        # Average feature importance
        if n_folds_with_importance > 0:
            full_results.feature_importance_scores = (
                full_results.feature_importance_scores / n_folds_with_importance
            )

        # Average learning curves
        if hasattr(full_results.learning_curve_data, 'train_scores') and n_folds_with_curves > 0:
            full_results.learning_curve_data.train_scores = (
                full_results.learning_curve_data.train_scores / n_folds_with_curves
            ).tolist()
            full_results.learning_curve_data.val_scores = (
                full_results.learning_curve_data.val_scores / n_folds_with_curves
            ).tolist()
            full_results.n_folds = n_folds_with_curves

        self.app_logger.structured_log(logging.INFO, "Cross-validation completed",
                                     final_feature_shape=full_results.feature_data.shape,
                                     final_shap_shape=full_results.shap_values.shape 
                                     if full_results.shap_values is not None else None)