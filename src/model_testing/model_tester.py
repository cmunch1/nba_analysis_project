import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Union, Any
import yaml
import os
from sklearn import tree, ensemble
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import mlflow
import mlflow.sklearn
from ..config.config import AbstractConfig
from ..logging.logging_utils import log_performance, structured_log
from ..error_handling.custom_exceptions import ModelTestingError
from .abstract_model_testing import AbstractModelTester
from ..data_access.data_access import AbstractDataAccess
from .data_classes import ModelTrainingResults, ClassificationMetrics
from .modular_preprocessor import ModularPreprocessor
import lightgbm as lgb

logger = logging.getLogger(__name__)

class ModelTester(AbstractModelTester):
    @log_performance
    def __init__(self, config: AbstractConfig):
        """
        Initialize the ModelTester class.

        Args:
            config (AbstractConfig): Configuration object containing model testing parameters.
        """
        self.config = config
        self.preprocessor = ModularPreprocessor(config)
        structured_log(logger, logging.INFO, "ModelTester initialized",
                      config_type=type(config).__name__)

    def get_model_params(self, model_name: str) -> Dict:
        """
        Get the current best hyperparameters for a model from the configuration.

        Args:
            model_name (str): Name of the model to get parameters for.

        Returns:
            Dict: Dictionary containing the model's hyperparameters.

        Raises:
            ModelTestingError: If there's an error retrieving the hyperparameters.
        """
        try:
            if not hasattr(self.config, 'model_hyperparameters') or not hasattr(self.config.model_hyperparameters, model_name):
                raise ValueError(f"Hyperparameters for {model_name} not found in config")

            hyperparameters_list = getattr(self.config.model_hyperparameters, model_name)
            current_best = next((config['params'] for config in hyperparameters_list if config['name'] == 'current_best'), None)
            
            if current_best is None:
                raise ValueError(f"'current_best' configuration not found for {model_name}")

            structured_log(logger, logging.INFO, f"Loaded 'current_best' hyperparameters for {model_name}")
            return current_best
        except Exception as e:
            raise ModelTestingError(f"Error getting hyperparameters for {model_name}",
                                  error_message=str(e))

    @log_performance
    def prepare_data(self, df: pd.DataFrame, model_name: str = None, is_training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare the data for model training or validation with preprocessing.

        Args:
            df (pd.DataFrame): Input dataframe.
            model_name (str): Name of the model to prepare data for.
            is_training (bool): Whether this is training data (fit_transform) or validation data (transform)

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Tuple containing the feature dataframe and the target series.
        """
        structured_log(logger, logging.INFO, "Starting data preparation",
                      input_shape=df.shape,
                      is_training=is_training)
        try:
            target = self.config.home_team_prefix + self.config.target_column
            y = df[target]
            X = df.drop(columns=[target])
                
            
            preprocessing_steps = ModelTrainingResults(X.shape)
            
            # Store original column names
            original_columns = X.columns.tolist()
            
            # Apply preprocessing with tracking
            if is_training:
                X, preprocessing_steps = self.preprocessor.fit_transform(
                    X,
                    y=y,
                    model_name=model_name,
                    results=preprocessing_steps
                )
            else:
                X = self.preprocessor.transform(
                    X,
                )
            
            # Ensure column names are preserved after preprocessing
            if isinstance(X, pd.DataFrame):
                if len(X.columns) == len(original_columns):
                    X.columns = original_columns
            else:
                X = pd.DataFrame(X, columns=original_columns)

            X = self._optimize_data_types(X)

            
            structured_log(logger, logging.INFO, "Data preparation completed",
                         output_shape=X.shape)
            return X, y, preprocessing_steps
        
        except Exception as e:
            raise ModelTestingError("Error in data preparation",
                                  error_message=str(e),
                                  dataframe_shape=df.shape)

    @log_performance
    def perform_oof_cross_validation(self, X: pd.DataFrame, y: pd.Series, model_name: str, model_params: Dict) -> ModelTrainingResults:
        """
        Perform Out-of-Fold (OOF) cross-validation with preprocessing tracking.
        """
        structured_log(logger, logging.INFO, f"{model_name} - Starting OOF cross-validation",
                      input_shape=X.shape)
        try:
            full_results = ModelTrainingResults(X.shape)
            full_results.predictions = np.zeros(len(y))
            full_results.shap_values = np.zeros((len(y), X.shape[1]))
            if self.config.calculate_shap_interactions:
                full_results.shap_interaction_values = np.zeros((len(y), X.shape[1], X.shape[1]))

            if self.config.cross_validation_type == "StratifiedKFold":
                kf = StratifiedKFold(n_splits=self.config.n_splits, shuffle=True, random_state=self.config.random_state)
            elif self.config.cross_validation_type == "TimeSeriesSplit":
                kf = TimeSeriesSplit(n_splits=self.config.n_splits)
            else:
                raise ValueError(f"Unsupported CV type: {self.config.cross_validation_type}")

            feature_importance_accumulator = np.zeros(X.shape[1])
            n_folds_with_importance = 0

            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Create fold-specific results object
                fold_results = ModelTrainingResults(X_train.shape)
                
                oof_results = self._train_model(
                    X_train, y_train,
                    X_val, y_val,
                    model_name, model_params
                )

                full_results.model = oof_results.model
                full_results.predictions[val_idx] = oof_results.predictions
                
                if oof_results.shap_values is not None:
                    if oof_results.shap_values.shape[1] == X.shape[1] + 1:
                        oof_shap_values = oof_results.shap_values[:, :-1]
                    else:
                        oof_shap_values = oof_results.shap_values
                    full_results.shap_values[val_idx] = oof_shap_values

                if oof_results.shap_interaction_values is not None and self.config.calculate_shap_interactions:
                    if oof_results.shap_interaction_values.shape[1] == X.shape[1] + 1:
                        oof_interaction_values = oof_results.shap_interaction_values[:, :-1, :-1]
                    else:
                        oof_interaction_values = oof_results.shap_interaction_values
                        
                    if oof_interaction_values.shape[1:] == full_results.shap_interaction_values.shape[1:]:
                        full_results.shap_interaction_values[val_idx] = oof_interaction_values
                    else:
                        structured_log(logger, logging.WARNING, 
                                    f"SHAP interaction values shape mismatch in fold {fold}")

                if oof_results.feature_importance_scores is not None:
                    feature_importance_accumulator += oof_results.feature_importance_scores
                    n_folds_with_importance += 1

                full_results.feature_names = oof_results.feature_names

                if oof_results.predictions is not None:
                    binary_predictions = (oof_results.predictions >= 0.5).astype(int)
                    fold_accuracy = accuracy_score(y_val, binary_predictions)
                    fold_auc = roc_auc_score(y_val, oof_results.predictions)
                    structured_log(logger, logging.INFO, f"Fold {fold} completed",
                                fold_accuracy=fold_accuracy, fold_auc=fold_auc)

            if n_folds_with_importance > 0:
                full_results.feature_importance_scores = feature_importance_accumulator / n_folds_with_importance

            overall_accuracy = accuracy_score(y, (full_results.predictions >= 0.5).astype(int))
            overall_auc = roc_auc_score(y, full_results.predictions)
            

            structured_log(logger, logging.INFO, "OOF cross-validation completed",
                        overall_accuracy=overall_accuracy, overall_auc=overall_auc)

            return full_results

        except Exception as e:
            raise ModelTestingError("Error in OOF cross-validation",
                                  error_message=str(e),
                                  dataframe_shape=X.shape)

    @log_performance
    def perform_validation_set_testing(self, X: pd.DataFrame, y: pd.Series, 
                                     X_val: pd.DataFrame, y_val: pd.Series, 
                                     model_name: str, model_params: Dict) -> ModelTrainingResults:
        """
        Perform model training and validation on a separate validation set.
        """
        try:
            # Create results object
            results = ModelTrainingResults(X.shape)
            
            results = self._train_model(
                X, y,
                X_val, y_val,
                model_name, model_params
            )
            
            return results
            
        except Exception as e:
            raise ModelTestingError("Error in validation set testing",
                                  error_message=str(e),
                                  dataframe_shape=X.shape)

    @log_performance
    def calculate_classification_evaluation_metrics(self, y_true, y_prob) -> ClassificationMetrics:
        """Calculate classification metrics using probability scores and optimal threshold."""
        metrics = ClassificationMetrics()
        
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
        metrics.optimal_threshold = optimal_threshold
        
        structured_log(logger, logging.INFO, "Classification metrics calculated",
                      optimal_threshold=optimal_threshold,
                      accuracy=metrics.accuracy,
                      auc=metrics.auc)
        
        return metrics

    @log_performance
    def _train_model(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, 
                     model_name: str, model_params: Dict) -> ModelTrainingResults:
        """Train a model on the given data."""
        structured_log(logger, logging.INFO, f"{model_name} - Starting model training",
                      input_shape=X.shape)
        try:
            
            print(X.columns)
            
            results = ModelTrainingResults(X.shape)
            
            results.feature_names = X.columns.tolist()
            results.update_feature_data(X_val, y_val)
            
            match model_name:
                case "XGBoost":
                    results = self._train_XGBoost(X, y, X_val, y_val, model_params)
                case "LGBM":
                    results = self._train_LGBM(X, y, X_val, y_val, model_params)
                case _:
                    results = self._train_sklearn_model(X, y, X_val, y_val, model_name, model_params)
            
            if results.predictions is None:
                raise ModelTestingError("Model training completed but predictions are None")
                                    
            structured_log(logger, logging.INFO, f"{model_name} - Model training completed",
                         predictions_shape=results.predictions.shape)
            
            return results
        
        except Exception as e:
            raise ModelTestingError("Error in model training",
                                  error_message=str(e),
                                  model_name=model_name,
                                  dataframe_shape=X.shape)

        
    @log_performance
    def _train_sklearn_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, model_name: str, model_params: Dict) -> ModelTrainingResults:
        """
        Train a scikit-learn model.

        Args:
            X_train (pd.DataFrame): Training feature dataframe.
            y_train (pd.Series): Training target series.
            X_val (pd.DataFrame): Validation feature dataframe.
            y_val (pd.Series): Validation target series.
            model_name (str): Name of the sklearn model (from tree or ensemble modules).
            model_params (Dict): Model hyperparameters.

        Returns:
            ModelTrainingResults: Object containing model, predictions, and various model analysis results.

        Raises:
            ModelTestingError: If there's an error during model training.
            ValueError: If the model name is not supported.
        """
        results = ModelTrainingResults(X_train.shape)

        # Ensure X_val has the same columns in the same order as X_train
        X_val = X_val[X_train.columns]

        structured_log(logger, logging.INFO, f"{model_name} - Starting model training",
                       input_shape=X_train.shape)
        try:
            # Get the appropriate model class
            if hasattr(tree, model_name):
                model_class = getattr(tree, model_name)
            elif hasattr(ensemble, model_name):
                model_class = getattr(ensemble, model_name)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            # Initialize and train the model
            model = model_class()
            model.set_params(**model_params)
            model.fit(X_train, y_train)
            
            # Store model and generate predictions
            results.model = model
            results.predictions = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val)

            # Safely get feature importance scores
            try:
                if hasattr(model, 'feature_importances_'):
                    results.feature_importance_scores = model.feature_importances_
                    results.feature_names = X_train.columns.tolist()
                    structured_log(logger, logging.INFO, "Feature importance scores calculated successfully",
                                num_features=len(results.feature_importance_scores))
            except Exception as e:
                structured_log(logger, logging.WARNING, "Failed to get feature importance scores",
                            error=str(e),
                            error_type=type(e).__name__)
                results.feature_importance_scores = None
                results.feature_names = None

            # Note: Most sklearn models don't support SHAP values directly
            results.shap_values = None
            results.shap_interaction_values = None
            
            structured_log(logger, logging.INFO, f"{model_name} - Predictions generated",
                          predictions_type=type(results.predictions).__name__,
                          predictions_shape=results.predictions.shape,
                          predictions_sample=results.predictions[:5].tolist())
            
            return results
        
        except Exception as e:
            raise ModelTestingError("Error in sklearn model training",
                                     error_message=str(e),
                                     dataframe_shape=X_train.shape)
        
    @log_performance
    def _train_XGBoost(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, model_params: Dict) -> ModelTrainingResults:
        """
        Train an XGBoost model with the enhanced ModelTrainingResults.
        Returns:
            ModelTrainingResults object with trained model and predictions
        """
        structured_log(logger, logging.INFO, "Starting XGBoost training", 
                    input_shape=X_train.shape)
        
        try:
            # Initialize results
            results = ModelTrainingResults(X_train.shape)
            results.model_name = "XGBoost"
            results.model_params = model_params
            results.feature_names = X_train.columns.tolist()
            results.update_feature_data(X_val, y_val)

            # Debug column alignment
            structured_log(logger, logging.DEBUG, "Column comparison",
                        train_columns=X_train.columns.tolist(),
                        val_columns=X_val.columns.tolist())

            # Instead of using column indexing, ensure columns match using merge/join
            missing_cols = set(X_train.columns) - set(X_val.columns)
            extra_cols = set(X_val.columns) - set(X_train.columns)
            
            if missing_cols or extra_cols:
                structured_log(logger, logging.WARNING, "Column mismatch detected",
                            missing_columns=list(missing_cols),
                            extra_columns=list(extra_cols))
                
                # Add missing columns with zeros
                for col in missing_cols:
                    X_val[col] = 0
                    
                # Remove extra columns
                X_val = X_val[X_train.columns]

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

            # Train the model
            model = xgb.train(
                params=model_params,
                dtrain=dtrain,
                num_boost_round=self.config.XGB.num_boost_round,
                early_stopping_rounds=self.config.XGB.early_stopping_rounds,
                evals=[(dtrain, 'train'), (dval, 'eval')],
                verbose_eval=self.config.XGB.verbose_eval
            )

            # Store model and generate predictions
            results.model = model
            results.predictions = model.predict(dval)
            
            structured_log(logger, logging.INFO, "Generated predictions", 
                        predictions_shape=results.predictions.shape,
                        predictions_mean=float(np.mean(results.predictions)))

            # Calculate feature importance
            try:
                importance_dict = model.get_score(importance_type='gain')
                results.feature_importance_scores = np.array([
                    importance_dict.get(feature, 0) 
                    for feature in X_train.columns
                ])
                
                structured_log(logger, logging.INFO, "Calculated feature importance",
                            num_features_with_importance=len(importance_dict))
                
            except Exception as e:
                structured_log(logger, logging.WARNING, "Failed to calculate feature importance",
                            error=str(e))
                results.feature_importance_scores = np.zeros(X_train.shape[1])

            # Calculate SHAP values if configured
            if hasattr(self.config, 'calculate_shap_values') and self.config.calculate_shap_values:
                try:
                    results.shap_values = model.predict(dval, pred_contribs=True)
                    structured_log(logger, logging.INFO, "Calculated SHAP values",
                                shap_values_shape=results.shap_values.shape)
                    
                    if self.config.calculate_shap_interactions:
                        n_features = X_val.shape[1]
                        estimated_memory = (X_val.shape[0] * n_features * n_features * 8) / (1024 ** 3)
                        
                        if estimated_memory <= self.config.max_shap_interaction_memory_gb:
                            results.shap_interaction_values = model.predict(dval, pred_interactions=True)
                            structured_log(logger, logging.INFO, "Calculated SHAP interaction values")
                            
                except Exception as e:
                    structured_log(logger, logging.WARNING, "Failed to calculate SHAP values",
                                error=str(e))
            
            # Verify predictions exist and are valid
            if results.predictions is None:
                raise ModelTestingError("XGBoost model.predict returned None")
            
            if len(results.predictions) != len(y_val):
                raise ModelTestingError(
                    "Prediction length mismatch",
                    expected_length=len(y_val),
                    actual_length=len(results.predictions)
                )
            
            structured_log(logger, logging.INFO, "XGBoost training completed successfully",
                        predictions_shape=results.predictions.shape)
            
            return results
    
        except Exception as e:
            structured_log(logger, logging.ERROR, "Error in XGBoost training",
                        error_message=str(e),
                        error_type=type(e).__name__,
                        dataframe_shape=X_train.shape)
            raise ModelTestingError(
                "Error in XGBoost model training",
                error_message=str(e),
                dataframe_shape=X_train.shape
            )
        
    @log_performance
    def _train_LGBM(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, model_params: Dict) -> ModelTrainingResults:
        """
        Train a LightGBM model with the enhanced ModelTrainingResults.
        Returns:
            ModelTrainingResults object with trained model and predictions
        """
        structured_log(logger, logging.INFO, "Starting LightGBM training", 
                    input_shape=X_train.shape)
        
        try:
            # Initialize results
            results = ModelTrainingResults(X_train.shape)
            results.model_name = "LightGBM"
            results.model_params = model_params
            results.feature_names = X_train.columns.tolist()
            results.update_feature_data(X_val, y_val)

            # Ensure X_val has the same columns
            X_val = X_val[X_train.columns]
            
            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=self.config.categorical_features)
            val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=self.config.categorical_features, reference=train_data)
            
            # Train the model
            model = lgb.train(
                model_params,
                train_data,
                num_boost_round=self.config.LGBM.num_boost_round,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(self.config.LGBM.early_stopping),
                    lgb.log_evaluation(self.config.LGBM.log_evaluation)
                ]
            )

            # Store model and generate predictions
            results.model = model
            results.predictions = model.predict(X_val)
            
            structured_log(logger, logging.INFO, "Generated predictions", 
                        predictions_shape=results.predictions.shape,
                        predictions_mean=float(np.mean(results.predictions)))

            # Calculate feature importance
            try:
                results.feature_importance_scores = model.feature_importance('gain')
                results.feature_names = X_train.columns.tolist()
                
                structured_log(logger, logging.INFO, "Calculated feature importance",
                            num_features_with_importance=len(results.feature_importance_scores))
                
            except Exception as e:
                structured_log(logger, logging.WARNING, "Failed to calculate feature importance",
                            error=str(e))
                results.feature_importance_scores = np.zeros(X_train.shape[1])

            # Calculate SHAP values if configured
            if hasattr(self.config, 'calculate_shap_values') and self.config.calculate_shap_values:
                try:
                    results.shap_values = model.predict(X_val, pred_contrib=True)
                    structured_log(logger, logging.INFO, "Calculated SHAP values",
                                shap_values_shape=results.shap_values.shape)
                    
                    if self.config.calculate_shap_interactions:
                        n_features = X_val.shape[1]
                        estimated_memory = (X_val.shape[0] * n_features * n_features * 8) / (1024 ** 3)
                        
                        if estimated_memory <= self.config.max_shap_interaction_memory_gb:
                            results.shap_interaction_values = model.predict(X_val, pred_interactions=True)
                            structured_log(logger, logging.INFO, "Calculated SHAP interaction values")
                            
                except Exception as e:
                    structured_log(logger, logging.WARNING, "Failed to calculate SHAP values",
                                error=str(e))
            
            # Verify predictions exist and are valid
            if results.predictions is None:
                raise ModelTestingError("LightGBM model.predict returned None")
            
            if len(results.predictions) != len(y_val):
                raise ModelTestingError(
                    "Prediction length mismatch",
                    expected_length=len(y_val),
                    actual_length=len(results.predictions)
                )
            
            structured_log(logger, logging.INFO, "LightGBM training completed successfully",
                        predictions_shape=results.predictions.shape)
            
            return results

        except Exception as e:
            structured_log(logger, logging.ERROR, "Error in LightGBM training",
                        error_message=str(e),
                        error_type=type(e).__name__,
                        dataframe_shape=X_train.shape)
            raise ModelTestingError(
                "Error in LightGBM model training",
                error_message=str(e),
                dataframe_shape=X_train.shape
            )
        
    @log_performance
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize data types of the dataframe and convert date field to proper date type.
        The primarily reduces the bitsize of ints and floats to reduce memory usage and improve performance.

        Args:
            df (pd.DataFrame): Input dataframe.
            date_column (str): Name of the date column. Defaults to 'date'.

        Returns:
            pd.DataFrame: Dataframe with optimized data types.
        """
        structured_log(logger, logging.INFO, "Starting data type optimization",
                       input_shape=df.shape, input_memory=df.memory_usage().sum() / 1e6)
        try:

            # Optimize numeric columns
            for col in df.select_dtypes(include=['int', 'float']).columns:
                col_min, col_max = df[col].min(), df[col].max()

                # For integer columns
                if df[col].dtype.kind in ['i', 'u']:
                    if col_min >= -128 and col_max <= 127:
                        df[col] = df[col].astype(np.int8)
                    elif col_min >= -32768 and col_max <= 32767:
                        df[col] = df[col].astype(np.int16)
                    elif col_min >= -2147483648 and col_max <= 2147483647:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)

                # For float columns
                else:
                    df[col] = df[col].astype(np.float16)

            structured_log(logger, logging.INFO, "Data type optimization completed",
                           output_shape=df.shape, output_memory=df.memory_usage().sum() / 1e6)
            return df
        except Exception as e:
            raise ModelTestingError("Error in data type optimization",
                                     error_message=str(e),
                                     dataframe_shape=df.shape)

