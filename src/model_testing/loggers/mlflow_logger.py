import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
import tempfile
import os
from mlflow.models.signature import infer_signature
from typing import Dict, Any
import numpy as np
import pandas as pd
from ..abstract_model_testing import AbstractExperimentLogger
from ...config.config import AbstractConfig
from ...logging.logging_utils import log_performance, structured_log
from ...error_handling.custom_exceptions import ChartCreationError
from ..chart_functions import ChartFunctions
from ..data_classes import ModelTrainingResults

logger = logging.getLogger(__name__)

class MLflowChartLogger:
    """Helper class for generating and logging charts to MLflow."""
    
    @log_performance
    def __init__(self):
        """Initialize the MLflow chart logger with ChartFunctions."""
        self.chart_functions = ChartFunctions()
        structured_log(logger, logging.INFO, "MLflowChartLogger initialized")

    @log_performance
    def log_model_charts(self, results: ModelTrainingResults) -> None:
        """
        Generate and log model evaluation charts to MLflow.
        
        Args:
            results: ModelTrainingResults object containing all necessary data
        """
        try:
            chart_data = results.prepare_for_charting()
            
            # Create temporary directory for saving charts
            with tempfile.TemporaryDirectory() as temp_dir:
                # Feature Importance Chart
                if chart_data["feature_importance"] is not None:
                    fig = self.chart_functions.create_feature_importance_chart(
                        feature_importance=chart_data["feature_importance"],
                        feature_names=chart_data["feature_names"]
                    )
                    self._save_and_log_figure(fig, f"{chart_data['prefix']}feature_importance", temp_dir)

                # Confusion Matrix
                if chart_data["y_true"] is not None and chart_data["y_pred"] is not None:
                    fig = self.chart_functions.create_confusion_matrix(
                        y_true=chart_data["y_true"],
                        y_pred=chart_data["y_pred"]
                    )
                    self._save_and_log_figure(fig, f"{chart_data['prefix']}confusion_matrix", temp_dir)

                # ROC Curve
                if chart_data["y_true"] is not None and chart_data["y_prob"] is not None:
                    fig = self.chart_functions.create_roc_curve(
                        y_true=chart_data["y_true"],
                        y_score=chart_data["y_prob"]
                    )
                    self._save_and_log_figure(fig, f"{chart_data['prefix']}roc_curve", temp_dir)

                # SHAP Summary Plot
                try:
                    if chart_data["model"] is not None and chart_data["X"] is not None:
                        fig = self.chart_functions.create_shap_summary_plot(
                            model=chart_data["model"],
                            X=chart_data["X"]
                        )
                        self._save_and_log_figure(fig, f"{chart_data['prefix']}shap_summary", temp_dir)
                except Exception as e:
                    structured_log(logger, logging.WARNING, 
                                 "Failed to create SHAP summary plot",
                                 error=str(e))

                # Learning Curve
                try:
                    if all(v is not None for v in [chart_data["model"], chart_data["X"], chart_data["y_true"]]):
                        fig = self.chart_functions.create_learning_curve(
                            model=chart_data["model"],
                            X=chart_data["X"].values,
                            y=chart_data["y_true"]
                        )
                        self._save_and_log_figure(fig, f"{chart_data['prefix']}learning_curve", temp_dir)
                except Exception as e:
                    structured_log(logger, logging.WARNING, 
                                 "Failed to create learning curve",
                                 error=str(e))

                structured_log(logger, logging.INFO, "All charts logged successfully")

        except Exception as e:
            raise ChartCreationError("Error in chart logging",
                                   error_message=str(e))

    def _save_and_log_figure(self, fig, name: str, temp_dir: str) -> None:
        """
        Save a figure to a temporary file and log it to MLflow.
        
        Args:
            fig: Matplotlib figure object
            name: Name for the figure
            temp_dir: Temporary directory path
        """
        if fig is not None:
            filepath = os.path.join(temp_dir, f"{name}.png")
            fig.savefig(filepath, bbox_inches='tight', dpi=300)
            mlflow.log_artifact(filepath, "charts")
            fig.clf()  # Clear the figure to free memory


class MLFlowLogger(AbstractExperimentLogger):
    def __init__(self, config: AbstractConfig):
        self.config = config
        self._chart_logger = MLflowChartLogger()
        structured_log(logger, logging.INFO, "MLFlowLogger initialized")
    
    def log_experiment(self, results: ModelTrainingResults):
        """Log an experiment to MLflow with preprocessing tracking."""
        if results.model is None:
            raise ValueError("Cannot log experiment: model object is None")

        with mlflow.start_run():
            mlflow.set_experiment(self.config.experiment_name)
            mlflow.set_tag("description", self.config.experiment_description)
            
            # Log model parameters
            mlflow.log_params(results.model_params)
            
            # Log preprocessing configuration and summary
            if results.preprocessing_config:
                mlflow.log_params({
                    "preprocessing": results.preprocessing_config
                })
            
            preprocessing_summary = results.summarize_preprocessing()
            mlflow.log_params({
                "n_original_features": preprocessing_summary["n_original_features"],
                "n_final_features": preprocessing_summary["n_final_features"],
                "n_preprocessing_steps": preprocessing_summary["n_preprocessing_steps"]
            })
            
            # Log detailed preprocessing information as artifacts
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save preprocessing info as JSON
                preproc_path = os.path.join(temp_dir, "preprocessing_info.json")
                with open(preproc_path, "w") as f:
                    json.dump(results.preprocessing_info.to_dict(), f, indent=2)
                mlflow.log_artifact(preproc_path, "preprocessing")
                
                # Save feature transformations
                transformations = results.get_feature_transformations()
                trans_path = os.path.join(temp_dir, "feature_transformations.json")
                with open(trans_path, "w") as f:
                    json.dump(transformations, f, indent=2)
                mlflow.log_artifact(trans_path, "preprocessing")
            
            try:
                # Prepare MLflow example data
                mlflow_example = results.feature_data.head(5).copy()
                for col in mlflow_example.select_dtypes(include=['int']).columns:
                    mlflow_example[col] = mlflow_example[col].astype('float64')
                
                signature = infer_signature(mlflow_example, results.target_data.head())
                
                # Log model using appropriate MLflow flavor
                if 'xgboost' in results.model_name.lower():
                    mlflow.xgboost.log_model(
                        results.model,
                        results.model_name,
                        signature=signature,
                        input_example=mlflow_example
                    )
                elif 'lgbm' in results.model_name.lower():
                    mlflow.lightgbm.log_model(
                        results.model,
                        results.model_name,
                        signature=signature,
                        input_example=mlflow_example
                    )
                else:
                    mlflow.sklearn.log_model(
                        results.model,
                        results.model_name,
                        signature=signature,
                        input_example=mlflow_example
                    )

                # Log charts
                self._chart_logger.log_model_charts(results)
                
                structured_log(logger, logging.INFO, "Experiment logged successfully", 
                             model_name=results.model_name)

            except Exception as e:
                structured_log(logger, logging.ERROR, "Failed to log experiment",
                             error=str(e))
                raise

    @log_performance
    def log_model(self, model: object, model_name: str, model_params: dict):
        """
        Log a model to MLflow with its parameters.
        
        Args:
            model: The trained model object
            model_name: Name of the model
            model_params: Dictionary of model parameters
        """
        try:
            with mlflow.start_run():
                mlflow.set_experiment(self.config.experiment_name)
                mlflow.log_params(model_params)
                
                # Log model using appropriate MLflow flavor
                if 'xgboost' in model_name.lower():
                    mlflow.xgboost.log_model(model, model_name)
                elif 'lgbm' in model_name.lower():
                    mlflow.lightgbm.log_model(model, model_name)
                else:
                    mlflow.sklearn.log_model(model, model_name)
                    
                structured_log(logger, logging.INFO, "Model logged successfully", 
                             model_name=model_name)
                             
        except Exception as e:
            raise ModelTestingError("Error logging model to MLflow",
                                  error_message=str(e),
                                  model_name=model_name)