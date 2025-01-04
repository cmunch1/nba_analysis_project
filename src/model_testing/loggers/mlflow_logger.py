import sys
import traceback
import logging
import mlflow
import mlflow.sklearn
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
        
        with mlflow.start_run():
            mlflow.set_experiment(experiment_name)
            mlflow.set_tag("description", experiment_description)
            mlflow.log_params(model_params)

            # Log model with signature and input example
            if 'oof_data' in kwargs:
                input_example = kwargs['oof_data'].drop('target', axis=1).head(5)
                signature = infer_signature(
                    kwargs['oof_data'].drop('target', axis=1),
                    kwargs['oof_data']['target']
                )
            elif 'val_data' in kwargs:
                input_example = kwargs['val_data'].drop('target', axis=1).head(5)
                signature = infer_signature(
                    kwargs['val_data'].drop('target', axis=1),
                    kwargs['val_data']['target']
                )

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
                mlflow.log_metrics({
                    "oof_" + k: v for k, v in kwargs['oof_metrics'].__dict__.items()
                })

            # Log validation results if present
            if 'val_metrics' in kwargs and 'val_data' in kwargs:
                val_dataset = mlflow.data.from_pandas(
                    kwargs['val_data'].drop(['val_predictions'], axis=1),
                    targets="target"
                )
                mlflow.log_input(val_dataset, context="validation")
                mlflow.log_metrics({
                    "val_" + k: v for k, v in kwargs['val_metrics'].__dict__.items()
                })


    def log_model(self, model, model_name: str, model_params: dict):
        mlflow.sklearn.log_model(model, model_name, params=model_params)    

