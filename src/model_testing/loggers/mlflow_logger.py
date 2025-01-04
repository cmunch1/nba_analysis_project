import sys
import traceback
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
import shap
import matplotlib.pyplot as plt
from ..abstract_model_testing import AbstractExperimentLogger
from ...config.config import AbstractConfig
from ...model_testing.data_classes import ClassificationMetrics

class MLFlowLogger(AbstractExperimentLogger):
    def __init__(self, config: AbstractConfig):
        self.config = config
    
    def log_experiment(self, 
                      experiment_name: str, 
                      experiment_description: str, 
                      model_name: str,
                      model: object, 
                      model_params: dict,
                      **kwargs):  # Changed to accept flexible kwargs
        
        # Add validation check for model
        if model is None:
            raise ValueError("Cannot log experiment: model object is None")
        
        # Check if model has required attributes based on type
        if 'xgboost' in model_name.lower():
            if not hasattr(model, 'save_model'):
                raise ValueError("XGBoost model missing save_model method. Check if model.model contains the actual model object")
            model_to_log = model.model if hasattr(model, 'model') else model
        else:
            model_to_log = model

        with mlflow.start_run():
            mlflow.set_experiment(experiment_name)
            mlflow.set_tag("description", experiment_description)
            mlflow.log_params(model_params)

            try:
                # Get the original data
                if 'oof_data' in kwargs:
                    input_df = kwargs['oof_data'].drop('target', axis=1)
                    target_series = kwargs['oof_data']['target']
                elif 'val_data' in kwargs:
                    input_df = kwargs['val_data'].drop('target', axis=1)
                    target_series = kwargs['val_data']['target']
                else:
                    raise ValueError("Neither oof_data nor val_data provided")
                
                # Create a copy just for MLflow input example
                mlflow_example = input_df.head(5).copy()
                
                # Convert integers to float64 ONLY for MLflow example
                for col in mlflow_example.select_dtypes(include=['int']).columns:
                    mlflow_example[col] = mlflow_example[col].astype('float64')
                
                signature = infer_signature(mlflow_example, target_series.head())
                
                # Log model using the float-converted example but keep original model intact
                if 'xgboost' in model_name.lower():
                    mlflow.xgboost.log_model(
                        model_to_log, 
                        model_name,
                        signature=signature,
                        input_example=mlflow_example
                    )
                elif 'lgbm' in model_name.lower():
                    mlflow.lightgbm.log_model(
                        model, 
                        model_name,
                        signature=signature,
                        input_example=input_example
                    )
                elif 'catboost' in model_name.lower():
                    mlflow.catboost.log_model(
                        model, 
                        model_name,
                        signature=signature,
                        input_example=input_example
                    )
                else:
                    mlflow.sklearn.log_model(
                        model, 
                        model_name,
                        signature=signature,
                        input_example=input_example
                    )

                # Log OOF results if present
                if 'oof_metrics' in kwargs and 'oof_data' in kwargs:
                    oof_dataset = mlflow.data.from_pandas(
                        kwargs['oof_data'].drop(['oof_predictions'], axis=1),
                        targets="target"
                    )
                    mlflow.log_input(oof_dataset, context="training")
                    # Convert metrics to standard Python floats
                    oof_metrics = {
                        "oof_" + k: float(v) for k, v in kwargs['oof_metrics'].__dict__.items()
                    }
                    mlflow.log_metrics(oof_metrics)

                # Log validation results if present
                if 'val_metrics' in kwargs and 'val_data' in kwargs:
                    val_dataset = mlflow.data.from_pandas(
                        kwargs['val_data'].drop(['val_predictions'], axis=1),
                        targets="target"
                    )
                    mlflow.log_input(val_dataset, context="validation")
                    # Convert metrics to standard Python floats
                    val_metrics = {
                        "val_" + k: float(v) for k, v in kwargs['val_metrics'].__dict__.items()
                    }
                    mlflow.log_metrics(val_metrics)

            except Exception as e:
                logging.error(f"Failed to log model: {str(e)}")
                raise


    def log_model(self, model, model_name: str, model_params: dict):
        mlflow.sklearn.log_model(model, model_name, params=model_params)    

