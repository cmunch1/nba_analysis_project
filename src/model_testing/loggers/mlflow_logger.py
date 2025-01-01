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


class MLFlowLogger(AbstractExperimentLogger):
    def __init__(self, config: AbstractConfig):
        self.config = config
    
    def log_experiment(self, experiment_name: str, experiment_description: str, model_name: str, model: object, model_params: dict, oof_metrics: dict, val_metrics: dict, oof_data: pd.DataFrame, val_data: pd.DataFrame):
        with mlflow.start_run():
            mlflow.set_experiment(experiment_name)
            mlflow.set_tag("description", experiment_description)
            mlflow.log_params(model_params)

            # Log model with signature and input example
            input_example = oof_data.drop('target', axis=1).head(5)
            signature = infer_signature(
                oof_data.drop('target', axis=1),
                oof_data['target']
            )
            mlflow.sklearn.log_model(
                model, 
                model_name,
                signature=signature,
                input_example=input_example
            )

            # Create datasets for evaluation
            oof_dataset = mlflow.data.from_pandas(
                oof_data.drop(['oof_predictions'], axis=1),  # Remove predictions column
                targets="target"
            )
            
            val_dataset = mlflow.data.from_pandas(
                val_data.drop(['val_predictions'], axis=1),  # Remove predictions column
                targets="target"
            )

            # Log the datasets
            mlflow.log_input(oof_dataset, context="training")
            mlflow.log_input(val_dataset, context="validation")

            # Log metrics directly instead of using evaluate
            mlflow.log_metrics(oof_metrics)
            mlflow.log_metrics(val_metrics)


    def log_model(self, model, model_name: str, model_params: dict):
        mlflow.sklearn.log_model(model, model_name, params=model_params)    

