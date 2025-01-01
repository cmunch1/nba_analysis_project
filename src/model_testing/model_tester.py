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
#from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from ..config.config import AbstractConfig
from ..logging.logging_utils import log_performance, structured_log
from ..error_handling.custom_exceptions import ModelTestingError
from .abstract_model_testing import AbstractModelTester
from ..data_access.data_access import AbstractDataAccess
from .data_classes import ModelTrainingResults, ClassificationMetrics
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
        
        structured_log(logger, logging.INFO, "ModelTester initialized",
                       config_type=type(config).__name__)
        

        
    @log_performance
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare the data for model training.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Tuple containing the feature dataframe and the target series.
        """ 

        structured_log(logger, logging.INFO, "Starting data preparation",
                       input_shape=df.shape)

        try:

            target = self.config.home_team_prefix + self.config.target_column # need to include home team prefix
            y = df[target]

            # drop target column 
            X = df.drop(columns=[target])

            # optimize the data types for memory and performance
            X = self._optimize_data_types(X)

            structured_log(logger, logging.INFO, "Data preparation completed",
                           output_shape=X.shape)    
            return X, y
        except Exception as e:
            raise ModelTestingError("Error in data preparation",
                                     error_message=str(e),
                                     dataframe_shape=df.shape)


    @log_performance
    def calculate_classification_evaluation_metrics(self, y_test: pd.Series, y_pred: pd.Series) -> ClassificationMetrics:
        """
        Calculate the evaluation metrics for binary classification predictions using an optimized threshold.

        Args:
            y_test (pd.Series): The true target values.
            y_pred (pd.Series): The predicted probabilities from the model.

        Returns:
            ClassificationMetrics: Dataclass containing evaluation metrics and optimal threshold
        """
        structured_log(logger, logging.INFO, "Calculating classification evaluation metrics")
        try:
            # Find optimal threshold by testing different thresholds
            # when converting probabilities to binary predictions, 
            # a threshold of 0.5 is often not the best threshold to convert to binary
            thresholds = np.arange(0.1, 0.9, 0.01)
            f1_scores = []
            
            for threshold in thresholds:
                y_pred_binary = (y_pred >= threshold).astype(int)
                f1 = f1_score(y_test, y_pred_binary, average='weighted')
                f1_scores.append(f1)
            
            # Get the threshold that maximizes F1 score
            optimal_threshold = thresholds[np.argmax(f1_scores)]
            
            # Use optimal threshold for final predictions
            y_pred_binary = (y_pred >= optimal_threshold).astype(int)
            
            metrics = ClassificationMetrics(
                accuracy=accuracy_score(y_test, y_pred_binary),
                precision=precision_score(y_test, y_pred_binary, average='weighted'),
                recall=recall_score(y_test, y_pred_binary, average='weighted'),
                f1=f1_score(y_test, y_pred_binary, average='weighted'),
                auc=roc_auc_score(y_test, y_pred),  # AUC uses probabilities directly
                optimal_threshold=float(optimal_threshold)
            )
            
            structured_log(logger, logging.INFO, "Classification evaluation metrics calculated", 
                         metrics=vars(metrics))
            return metrics
        except Exception as e:
            raise ModelTestingError("Error in model evaluation",
                                  error_message=str(e),
                                  dataframe_shape=y_test.shape)

 
    @log_performance
    def perform_oof_cross_validation(self, X: pd.DataFrame, y: pd.Series, model_name: str, model_params: Dict) -> ModelTrainingResults:
        """
        Perform Out-of-Fold (OOF) cross-validation on the data.

        This method performs k-fold cross-validation while retaining predictions for each fold,
        allowing for detailed model evaluation later (e.g., SHAP values, error analysis).
        Supports both StratifiedKFold and TimeSeriesSplit cross-validation strategies.

        Args:
            X (pd.DataFrame): The feature dataframe.
            y (pd.Series): The target series.
            model_name (str): Name of the model to use ('XGBoost', 'LGBM', or sklearn model names).
            model_params (Dict): Model hyperparameters.

        Returns:
            ModelTrainingResults: Dataclass containing OOF predictions, SHAP values, and feature importances.
        """
        structured_log(logger, logging.INFO, f"{model_name} - Starting OOF cross-validation",
                       input_shape=X.shape)
        try:
            full_results = ModelTrainingResults(X.shape)
            oof_results = ModelTrainingResults(X.shape)

            # Choose CV strategy
            if self.config.cross_validation_type == "StratifiedKFold":
                kf = StratifiedKFold(n_splits=self.config.n_splits, shuffle=True, random_state=self.config.random_state)
            elif self.config.cross_validation_type == "TimeSeriesSplit":
                kf = TimeSeriesSplit(n_splits=self.config.n_splits)
            else:
                raise ValueError(f"Unsupported CV type: {self.config.cross_validation_type}")

            # Perform k-fold cross-validation
            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                oof_results = self._train_model(X_train, y_train, X_val, y_val, model_name, model_params)
                
                # Store results
                full_results.predictions[val_idx] = oof_results.predictions
                
                
                # Calculate fold metrics using 0.5 threshold for simple progress tracking
                binary_predictions = (oof_results.predictions >= 0.5).astype(int)
                fold_accuracy = accuracy_score(y_val, binary_predictions)
                fold_auc = roc_auc_score(y_val, oof_results.predictions)

                structured_log(logger, logging.INFO, f"Fold {fold} completed",
                               fold_accuracy=fold_accuracy, fold_auc=fold_auc)

            # Calculate overall metrics
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
    def perform_validation_set_testing(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, model_name: str, model_params: Dict) -> Tuple[Any, pd.Series]:
        """
        Perform model training and validation on a separate validation set.

        Args:
            X (pd.DataFrame): Training feature dataframe.
            y (pd.Series): Training target series.
            X_val (pd.DataFrame): Validation feature dataframe.
            y_val (pd.Series): Validation target series.
            model_name (str): Name of the model to use.
            model_params (Dict): Model hyperparameters.

        Returns:
            Tuple[Any, pd.Series]: Tuple containing:
                - The trained model
                - Series of predictions on validation set
        """
        full_results = ModelTrainingResults(X.shape)

        full_results = self._train_model(X, y, X_val, y_val, model_name, model_params)
           
        return full_results
    
    @log_performance
    def _train_model(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, model_name: str, model_params: Dict) -> Tuple[Any, np.ndarray]:
        """
        Train a model on the given data.

        Args:
            X (pd.DataFrame): Training feature dataframe.
            y (pd.Series): Training target series.
            X_val (pd.DataFrame): Validation feature dataframe.
            y_val (pd.Series): Validation target series.
            model_name (str): Name of the model to train ('XGBoost', 'LGBM', or sklearn model names).
            model_params (Dict): Model hyperparameters.

        Returns:
  
            ModelTrainingResults: Dataclass containing OOF predictions, SHAP values, and feature importances.

        Raises:
            ModelTestingError: If there's an error during model training.
        """
        structured_log(logger, logging.INFO, f"{model_name} - Starting model training",
                       input_shape=X.shape)
        try:
            results = ModelTrainingResults(X.shape)
            
            match model_name:
                case "XGBoost":
                    results = self._train_XGBoost(X, y, X_val, y_val, model_params)                         
                case "LGBM":
                    results = self._train_LGBM(X, y, X_val, y_val, model_params)
                case "CatBoost":
                    pass
                case _:
                    # then it is probably a Scikit-Learn model
                    try:
                        results = self._train_sklearn_model(X, y, X_val, y_val,model_name, model_params)
                    except Exception as e:
                        raise ModelTestingError(f"Error in model training",
                                                 error_message=str(e),
                                                 dataframe_shape=X.shape)
            
            structured_log(logger, logging.INFO, f"{model_name} - Model training completed")
            
            return results
        
        except Exception as e:
            raise ModelTestingError("Error in model training",
                                     error_message=str(e),
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
        Train an XGBoost model.

        Args:
            X_train (pd.DataFrame): Training feature dataframe.
            y_train (pd.Series): Training target series.
            X_val (pd.DataFrame): Validation feature dataframe.
            y_val (pd.Series): Validation target series.
            model_params (Dict): Model hyperparameters.

        Returns:
            ModelTrainingResults: Object containing model, predictions, and various model analysis results.
        """
        results = ModelTrainingResults(X_train.shape)

        # Ensure X_val has the same columns in the same order as X_train
        X_val = X_val[X_train.columns]
        
        train_dmatrix = xgb.DMatrix(X_train, label=y_train, enable_categorical=self.config.enable_categorical)
        val_dmatrix = xgb.DMatrix(X_val, label=y_val, enable_categorical=self.config.enable_categorical)

        # Create evaluation list for training
        evals = [(train_dmatrix, 'train'), (val_dmatrix, 'eval')]
        
        # Only use early stopping if configured
        early_stopping_rounds = self.config.XGB.early_stopping_rounds if hasattr(self.config, 'early_stopping_rounds') else None
        
        try:
            model = xgb.train(
                model_params, 
                train_dmatrix, 
                num_boost_round=self.config.XGB.num_boost_round,
                early_stopping_rounds=self.config.XGB.early_stopping_rounds,
                evals=evals,
            callbacks=[],
            verbose_eval=self.config.XGB.verbose_eval
            )
        except Exception as e:
            raise ModelTestingError("Error in model training",
                                     error_message=str(e),
                                     dataframe_shape=X_train.shape)

        # populate the results with all the data
        results.predictions = model.predict(val_dmatrix)
        results.model = model

        # Safely calculate SHAP values with error handling
        try:
            results.shap_values = model.predict(val_dmatrix, pred_contribs=True)
            structured_log(logger, logging.INFO, "SHAP values calculated successfully",
                        shape=results.shap_values.shape)
        except Exception as e:
            structured_log(logger, logging.WARNING, "Failed to calculate SHAP values",
                        error=str(e),
                        error_type=type(e).__name__)
            results.shap_values = None

        # Safely calculate SHAP interaction values
        try:
            if self.config.calculate_shap_interactions:
                # SHAP interactions can be memory-intensive for large datasets
                # Check if dataset size is reasonable before proceeding
                n_features = X_val.shape[1]
                estimated_memory = (X_val.shape[0] * n_features * n_features * 8) / (1024 ** 3)  # in GB
                
                if estimated_memory > self.config.max_shap_interaction_memory_gb:
                    raise ValueError(f"SHAP interaction calculation would require approximately "
                                f"{estimated_memory:.2f}GB memory, exceeding the limit of "
                                f"{self.config.max_shap_interaction_memory_gb}GB")
                    
                results.shap_interaction_values = model.predict(X_val, pred_interactions=True)
                structured_log(logger, logging.INFO, "SHAP interaction values calculated successfully",
                            shape=results.shap_interaction_values.shape)
            
        except Exception as e:
            structured_log(logger, logging.WARNING, "Failed to calculate SHAP interaction values",
                        error=str(e),
                        error_type=type(e).__name__)
            results.shap_interaction_values = None

        # Safely get feature importance scores
        try:
            feature_importance_dict = model.get_score(importance_type='gain')
            results.feature_importance_scores = np.array([feature_importance_dict.get(f, 0) for f in X_train.columns])
            results.feature_names = X_train.columns.tolist()
            structured_log(logger, logging.INFO, "Feature importance scores calculated successfully",
                        num_features=len(results.feature_importance_scores))
        except Exception as e:
            structured_log(logger, logging.WARNING, "Failed to get feature importance scores",
                        error=str(e),
                        error_type=type(e).__name__)
            results.feature_importance_scores = None
            results.feature_names = None
             
        
        structured_log(logger, logging.INFO, "XGBoost predictions generated",
                      predictions_type=type(results.predictions).__name__,
                      predictions_shape=results.predictions.shape,
                      predictions_sample=results.predictions[:5].tolist())
        
        return results
    
    @log_performance
    def _train_LGBM(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, model_params: Dict) -> ModelTrainingResults:
        """
        Train a LightGBM model.

        Args:
            X_train (pd.DataFrame): Training feature dataframe.
            y_train (pd.Series): Training target series.
            X_val (pd.DataFrame): Validation feature dataframe.
            y_val (pd.Series): Validation target series.
            model_params (Dict): Model hyperparameters.

        Returns:
            ModelTrainingResults: Object containing model, predictions, and various model analysis results.
        """
        results = ModelTrainingResults(X_train.shape)

        # Ensure X_val has the same columns in the same order as X_train
        X_val = X_val[X_train.columns]
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=self.config.categorical_features)
        val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=self.config.categorical_features, reference=train_data)
        
        # Create evaluation list for training
        evals = {'valid': val_data}
        
        # Only use early stopping if configured
        early_stopping_rounds = self.config.LGBM.early_stopping_rounds if hasattr(self.config, 'early_stopping_rounds') else None
        
        try:
            model = lgb.train(
                model_params,
                train_data,
                num_boost_round=self.config.LGBM.num_boost_round,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                early_stopping_rounds=self.config.LGBM.early_stopping_rounds,
                verbose_eval=self.config.LGBM.verbose_eval
            )
        except Exception as e:
            raise ModelTestingError("Error in LGBM model training",
                                  error_message=str(e),
                                  dataframe_shape=X_train.shape)

        # populate the results with all the data
        results.predictions = model.predict(X_val)
        results.model = model

        # Safely calculate SHAP values with error handling
        try:
            results.shap_values = model.predict(X_val, pred_contrib=True)
            structured_log(logger, logging.INFO, "SHAP values calculated successfully",
                         shape=results.shap_values.shape)
        except Exception as e:
            structured_log(logger, logging.WARNING, "Failed to calculate SHAP values",
                         error=str(e),
                         error_type=type(e).__name__)
            results.shap_values = None

        # Safely calculate SHAP interaction values
        try:
            if self.config.calculate_shap_interactions:
                # SHAP interactions can be memory-intensive for large datasets
                # Check if dataset size is reasonable before proceeding
                n_features = X_val.shape[1]
                estimated_memory = (X_val.shape[0] * n_features * n_features * 8) / (1024 ** 3)  # in GB
                
                if estimated_memory > self.config.max_shap_interaction_memory_gb:
                    raise ValueError(f"SHAP interaction calculation would require approximately "
                                  f"{estimated_memory:.2f}GB memory, exceeding the limit of "
                                  f"{self.config.max_shap_interaction_memory_gb}GB")
                    
                results.shap_interaction_values = model.predict(X_val, pred_interactions=True)
                structured_log(logger, logging.INFO, "SHAP interaction values calculated successfully",
                             shape=results.shap_interaction_values.shape)
            
        except Exception as e:
            structured_log(logger, logging.WARNING, "Failed to calculate SHAP interaction values",
                         error=str(e),
                         error_type=type(e).__name__)
            results.shap_interaction_values = None

        # Safely get feature importance scores
        try:
            results.feature_importance_scores = model.feature_importance('gain')
            results.feature_names = X_train.columns.tolist()
            structured_log(logger, logging.INFO, "Feature importance scores calculated successfully",
                         num_features=len(results.feature_importance_scores))
        except Exception as e:
            structured_log(logger, logging.WARNING, "Failed to get feature importance scores",
                         error=str(e),
                         error_type=type(e).__name__)
            results.feature_importance_scores = None
            results.feature_names = None
             
        structured_log(logger, logging.INFO, "LightGBM predictions generated",
                      predictions_type=type(results.predictions).__name__,
                      predictions_shape=results.predictions.shape,
                      predictions_sample=results.predictions[:5].tolist())
        
        return results
    
    @log_performance
    def _optimize_data_types(self, df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
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

    def get_model_params(self, model_name: str) -> Dict:
        """
        Get the current best hyperparameters for a model from the configuration.

        Args:
            model_name (str): Name of the model to get parameters for.

        Returns:
            Dict: Dictionary containing the model's hyperparameters.

        Raises:
            ModelTestingError: If there's an error retrieving the hyperparameters.
            ValueError: If hyperparameters for the model are not found in config or if
                       'current_best' configuration is not found.
        """
        try:
            if not hasattr(self.config, 'model_hyperparameters') or not hasattr(self.config.model_hyperparameters, model_name):
                raise ValueError(f"Hyperparameters for {model_name} not found in config")

            # Find the 'current_best' configuration
            hyperparameters_list = getattr(self.config.model_hyperparameters, model_name)
            current_best = next((config['params'] for config in hyperparameters_list if config['name'] == 'current_best'), None)
            
            if current_best is None:
                raise ValueError(f"'current_best' configuration not found for {model_name}")

            structured_log(logger, logging.INFO, f"Loaded 'current_best' hyperparameters for {model_name}")
            return current_best
        except Exception as e:
            raise ModelTestingError(f"Error getting hyperparameters for {model_name}",
                                     error_message=str(e))
