import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Union
import yaml
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from ..config.config import AbstractConfig
from ..logging.logging_utils import log_performance, structured_log
from ..error_handling.custom_exceptions import ModelTrainingError
from .abstract_model_training import AbstractModelTrainer

logger = logging.getLogger(__name__)

class ModelTrainer(AbstractModelTrainer):
    @log_performance
    def __init__(self, config: AbstractConfig):
        """
        Initialize the ModelTrainer class.

        Args:
            config (AbstractConfig): Configuration object containing model training parameters.
        """
        self.config = config
        structured_log(logger, logging.INFO, "ModelTrainer initialized",
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
            raise ModelTrainingError("Error in data preparation",
                                     error_message=str(e),
                                     dataframe_shape=df.shape)


    @log_performance
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the trained model.

        Args:
            model: The trained model object.
            X_test (pd.DataFrame): The test feature dataframe.
            y_test (pd.Series): The test target series.

        Returns:
            dict: Dictionary of evaluation metrics.
        """
        structured_log(logger, logging.INFO, "Starting model evaluation")
        try:
            y_pred = model.predict(X_test)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
            structured_log(logger, logging.INFO, "Model evaluation completed", metrics=metrics)
            return metrics
        except Exception as e:
            raise ModelTrainingError("Error in model evaluation",
                                     error_message=str(e),
                                     dataframe_shape=X_test.shape)

 

    @log_performance
    def perform_cross_validation(self, X: pd.DataFrame, y: pd.Series, model_name: str, model: Union[XGBClassifier, LGBMClassifier, RandomForestClassifier], cv_type: str = "StratifiedKFold", n_splits: int = 5) -> Dict[str, np.ndarray]:
        """
        Perform Out-of-Fold (OOF) cross-validation on the data.

        Args:
            X (pd.DataFrame): The feature dataframe.
            y (pd.Series): The target series.
            model (Union[XGBClassifier, RandomForestClassifier]): The model to use for cross-validation.
            cv_type (str): Type of cross-validation ('StratifiedKFold' or 'TimeSeriesSplit').
            n_splits (int): Number of cross-validation folds.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing OOF predictions, scores, and feature importances.
        """
        structured_log(logger, logging.INFO, f"{model_name} - Starting OOF cross-validation",
                       input_shape=X.shape, cv_type=cv_type, n_splits=n_splits)
        try:
            # Initialize OOF arrays
            oof_predictions = np.zeros(X.shape[0])
            oof_probabilities = np.zeros(X.shape[0])
            feature_importance = np.zeros(X.shape[1])

            # Choose CV strategy
            if cv_type == "StratifiedKFold":
                kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_state)
            elif cv_type == "TimeSeriesSplit":
                kf = TimeSeriesSplit(n_splits=n_splits)
            else:
                raise ValueError(f"Unsupported CV type: {cv_type}")

            # Perform k-fold cross-validation
            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train, y_train)

                oof_probabilities[val_idx] = model.predict_proba(X_val)[:, 1]
                oof_predictions[val_idx] = model.predict(X_val)
                feature_importance += model.feature_importances_

                fold_accuracy = accuracy_score(y_val, oof_predictions[val_idx])
                fold_auc = roc_auc_score(y_val, oof_probabilities[val_idx])

                structured_log(logger, logging.INFO, f"Fold {fold} completed",
                               fold_accuracy=fold_accuracy, fold_auc=fold_auc)

            # Calculate overall metrics
            overall_accuracy = accuracy_score(y, oof_predictions)
            overall_auc = roc_auc_score(y, oof_probabilities)

            feature_importance /= n_splits  # Average feature importance

            structured_log(logger, logging.INFO, "OOF cross-validation completed",
                           overall_accuracy=overall_accuracy, overall_auc=overall_auc)

            return {
                "oof_predictions": oof_predictions,
                "oof_probabilities": oof_probabilities,
                "feature_importance": feature_importance,
                "overall_accuracy": overall_accuracy,
                "overall_auc": overall_auc
            }

        except Exception as e:
            raise ModelTrainingError("Error in OOF cross-validation",
                                     error_message=str(e),
                                     dataframe_shape=X.shape)

    @log_performance
    def _optimize_data_types(self, df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """
        Optimize data types of the dataframe and convert date field to proper date type.

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
            raise ModelTrainingError("Error in data type optimization",
                                     error_message=str(e),
                                     dataframe_shape=df.shape)

    def get_model_params(self, model_name: str) -> Tuple[Union[XGBClassifier, LGBMClassifier, RandomForestClassifier], Dict]:
        """
        Get the model object and its hyperparameters from the config.

        Args:
            model_name (str): Name of the model to get parameters for.

        Returns:
            Tuple[Union[XGBClassifier, LGBMClassifier, RandomForestClassifier], Dict]: Tuple containing the model object and its hyperparameters.
        """
        try:
            if model_name == 'XGBClassifier':
                model_class = XGBClassifier
            elif model_name == 'LGBMClassifier':
                model_class = LGBMClassifier
            elif model_name == 'RandomForestClassifier':    
                model_class = RandomForestClassifier
            else:
                raise ValueError(f"Model {model_name} not supported")

            if not hasattr(self.config, 'model_hyperparameters') or model_name not in self.config.model_hyperparameters:
                raise ValueError(f"Hyperparameters for {model_name} not found in config")

            hyperparameters = self.config.model_hyperparameters[model_name]

            structured_log(logger, logging.INFO, f"Hyperparameters for {model_name} loaded successfully from config")
            return model_class(**hyperparameters), hyperparameters
        except Exception as e:
            raise ModelTrainingError(f"Error getting hyperparameters for {model_name}",
                                     error_message=str(e))
