import sys
import traceback
import logging
import mlflow
import mlflow.sklearn
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

            mlflow.sklearn.log_model(model, model_name)

            # Create the PandasDataset for use in mlflow evaluate
            pd_dataset = mlflow.data.from_pandas(
                oof_data, predictions="oof_predictions", targets="target"
            )

            pd_dataset = mlflow.data.from_pandas(
                val_data, predictions="val_predictions", targets="target"
            )

            mlflow.log_input(pd_dataset, context="validation")

            oof_result = mlflow.evaluate(
                data=oof_data,
                targets="target",
                predictions="oof_predictions",
                model_type="classifier",
            )

            validation_result = mlflow.evaluate(
                data=val_data,
                targets="target",
                predictions="val_predictions",
                model_type="classifier",
            )

            mlflow.log_metrics(oof_metrics)

            mlflow.log_metrics(validation_metrics)


    def log_model(self, model, model_name: str, model_params: dict):
        mlflow.sklearn.log_model(model, model_name, params=model_params)    

