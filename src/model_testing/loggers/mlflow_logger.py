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
import json

from ..abstract_model_testing import AbstractExperimentLogger
from ...config.config import AbstractConfig
from ...logging.logging_utils import log_performance, structured_log
from ...error_handling.custom_exceptions import ChartCreationError, ModelTestingError
from ..chart_functions import ChartFunctions
from ..data_classes import ModelTrainingResults

logger = logging.getLogger(__name__)

class MLflowChartLogger:
    """Helper class for generating and logging charts to MLflow."""
    
    @log_performance
    def __init__(self, config: AbstractConfig):
        """Initialize the MLflow chart logger with ChartFunctions."""
        self.config = config
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
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Feature Importance Chart
                if (hasattr(self.config.chart_options, 'feature_importance') and 
                    getattr(self.config.chart_options.feature_importance, 'enabled', False)):
                    try:    
                        if chart_data["feature_importance"] is not None:
                            fig = self.chart_functions.create_feature_importance_chart(
                                feature_importance=chart_data["feature_importance"],
                                feature_names=chart_data["feature_names"],
                                top_n=self.config.chart_options.feature_importance.n_features
                            )
                            self._save_and_log_figure(fig, f"{chart_data['prefix']}feature_importance", temp_dir)
                    except Exception as e:
                        structured_log(logger, logging.WARNING, 
                                     "Failed to create feature importance chart",
                                     error=str(e))

                # Confusion Matrix
                if self.config.chart_options.confusion_matrix:
                    try:
                        if chart_data["y_true"] is not None and chart_data["y_pred"] is not None:
                            fig = self.chart_functions.create_confusion_matrix(
                                y_true=chart_data["y_true"],
                                y_pred=chart_data["y_pred"]
                            )
                            self._save_and_log_figure(fig, f"{chart_data['prefix']}confusion_matrix", temp_dir)
                    except Exception as e:
                        structured_log(logger, logging.WARNING, 
                                     "Failed to create confusion matrix",
                                     error=str(e))

                # ROC Curve
                if self.config.chart_options.roc_curve:
                    try:
                        if chart_data["y_true"] is not None and chart_data["y_prob"] is not None:
                            fig = self.chart_functions.create_roc_curve(
                                y_true=chart_data["y_true"],
                                y_score=chart_data["y_prob"]
                                )
                            self._save_and_log_figure(fig, f"{chart_data['prefix']}roc_curve", temp_dir)
                    except Exception as e:
                        structured_log(logger, logging.WARNING, 
                                     "Failed to create ROC curve",
                                     error=str(e))

                # SHAP Summary Plot
                if (hasattr(self.config.chart_options, 'shap_summary') and 
                    getattr(self.config.chart_options.shap_summary, 'enabled', False)):
                    try:
                        if chart_data["model"] is not None and chart_data["X"] is not None:
                            fig = self.chart_functions.create_shap_summary_plot(
                                model=chart_data["model"],
                                X=chart_data["X"],
                                shap_values=results.shap_values,
                                n_features=self.config.chart_options.shap_summary.n_features
                            )
                            self._save_and_log_figure(fig, f"{chart_data['prefix']}shap_summary", temp_dir)
                    except Exception as e:
                        structured_log(logger, logging.WARNING, 
                                     "Failed to create SHAP summary plot",
                                     error=str(e))

                # SHAP Force Plot
                if (hasattr(self.config.chart_options, 'shap_force') and 
                    getattr(self.config.chart_options.shap_force, 'enabled', False)):
                    try:
                        if chart_data["model"] is not None and chart_data["X"] is not None:
                            # Limit features if specified
                            n_features = self.config.chart_options.shap_force.n_features
                            if n_features:
                                X_subset = chart_data["X"].iloc[:, :n_features]
                                shap_values = results.shap_values[:, :n_features] if results.shap_values is not None else None
                            else:
                                X_subset = chart_data["X"]
                                shap_values = results.shap_values

                            fig = self.chart_functions.create_shap_force_plot(
                                model=chart_data["model"],
                                X=X_subset,
                                shap_values=shap_values
                            )
                            self._save_and_log_figure(fig, f"{chart_data['prefix']}shap_force", temp_dir)
                    except Exception as e:
                        structured_log(logger, logging.WARNING, 
                                     "Failed to create SHAP force plot",
                                     error=str(e))

                # SHAP Dependence Plot
                if (hasattr(self.config.chart_options, 'shap_dependence') and 
                    getattr(self.config.chart_options.shap_dependence, 'enabled', False)):
                    try:
                        if chart_data["X"] is not None and results.shap_values is not None:
                            fig = self.chart_functions.create_shap_dependence_plot(
                                shap_values=results.shap_values,
                                features=chart_data["X"],
                                feature_name=chart_data["feature_names"][0]  # Using first feature as example
                            )
                            self._save_and_log_figure(fig, f"{chart_data['prefix']}shap_dependence", temp_dir)
                    except Exception as e:
                        structured_log(logger, logging.WARNING, 
                                     "Failed to create SHAP dependence plot",
                                     error=str(e))

                # Learning Curve
                if getattr(self.config.chart_options, 'learning_curve', False):
                    try:
                        fig = self.chart_functions.create_learning_curve(results=results)
                        self._save_and_log_figure(fig, f"{chart_data['prefix']}learning_curve", temp_dir)
                    except Exception as e:
                        structured_log(logger, logging.WARNING, 
                                     "Failed to create learning curve",
                                     error=str(e))

                # SHAP Waterfall Plot
                if (hasattr(self.config.chart_options, 'shap_waterfall') and 
                    getattr(self.config.chart_options.shap_waterfall, 'enabled', False)):
                    try:
                        if chart_data["model"] is not None and chart_data["X"] is not None:
                            # Limit features if specified
                            n_features = self.config.chart_options.shap_waterfall.n_features
                            if n_features:
                                X_subset = chart_data["X"].iloc[:, :n_features]
                                shap_values = results.shap_values[:, :n_features] if results.shap_values is not None else None
                            else:
                                X_subset = chart_data["X"]
                                shap_values = results.shap_values

                            fig = self.chart_functions.create_shap_waterfall_plot(
                                model=chart_data["model"],
                                X=X_subset,
                                shap_values=shap_values
                            )
                            self._save_and_log_figure(fig, f"{chart_data['prefix']}shap_waterfall", temp_dir)
                    except Exception as e:
                        structured_log(logger, logging.WARNING, 
                                    "Failed to create SHAP waterfall plot",
                                    error=str(e))

                # SHAP Beeswarm Plot
                if (hasattr(self.config.chart_options, 'shap_beeswarm') and 
                    getattr(self.config.chart_options.shap_beeswarm, 'enabled', False)):
                    try:
                        if chart_data["model"] is not None and chart_data["X"] is not None:
                            fig = self.chart_functions.create_shap_beeswarm_plot(
                                model=chart_data["model"],
                                X=chart_data["X"],
                                shap_values=results.shap_values,
                                n_features=self.config.chart_options.shap_beeswarm.n_features
                            )
                            self._save_and_log_figure(fig, f"{chart_data['prefix']}shap_beeswarm", temp_dir)
                    except Exception as e:
                        structured_log(logger, logging.WARNING, 
                                    "Failed to create SHAP beeswarm plot",
                                    X_shape=chart_data["X"].shape,
                                    shap_values_shape=results.shap_values.shape,
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
        self._chart_logger = MLflowChartLogger(self.config)
        structured_log(logger, logging.INFO, "MLFlowLogger initialized")
    
    def log_experiment(self, results: ModelTrainingResults):
        """Log an experiment to MLflow with preprocessing tracking."""
        if results.model is None:
            raise ValueError("Cannot log experiment: model object is None")

        # Set experiment before starting the run
        mlflow.set_experiment(self.config.experiment_name)
        
        # Create a unique run name based on evaluation type
        run_name = f"{results.model_name}_{results.evaluation_type}"
        
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("description", self.config.experiment_description)
            mlflow.set_tag("run_type", results.evaluation_type)  # Add run type tag
            
            # Log model parameters
            mlflow.log_params(results.model_params)

            # log additional parameters
            mlflow.log_params({
                "num_boost_round": results.num_boost_round,
                "early_stopping": results.early_stopping,
                "enable_categorical": results.enable_categorical,
                "categorical_features": results.categorical_features
            })
            
            # Log classification metrics
            if results.metrics:
                prefix = "val_" if results.is_validation else "oof_"
                mlflow.log_metrics({
                    f"{prefix}accuracy": results.metrics.accuracy,
                    f"{prefix}precision": results.metrics.precision,
                    f"{prefix}recall": results.metrics.recall,
                    f"{prefix}f1": results.metrics.f1,
                    f"{prefix}auc": results.metrics.auc,
                    f"{prefix}optimal_threshold": results.metrics.optimal_threshold
                })
            
            # Log preprocessing configuration and summary
            if results.preprocessing_results:
                mlflow.log_params({
                    "preprocessing": results.preprocessing_results.to_dict()
                })
            
            preprocessing_summary = results.preprocessing_results.summarize()
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
                    json.dump(results.preprocessing_results.to_dict(), f, indent=2)
                mlflow.log_artifact(preproc_path, "preprocessing")
                
                # Save feature transformations
                transformations = results.preprocessing_results.feature_transformations
                trans_path = os.path.join(temp_dir, "feature_transformations.json")
                with open(trans_path, "w") as f:
                    json.dump(transformations, f, indent=2)
                mlflow.log_artifact(trans_path, "preprocessing")
            
            # Log data shape information
            if results.feature_data is not None:
                mlflow.log_params({
                    "n_samples": results.feature_data.shape[0],
                    "n_features": results.feature_data.shape[1]
                })

            # Log evaluation context
            mlflow.log_params({
                "evaluation_type": results.evaluation_type,
                "is_validation": results.is_validation
            })

            # Log SHAP values summary if available
            if results.shap_values is not None:
                with tempfile.TemporaryDirectory() as temp_dir:
                    shap_path = os.path.join(temp_dir, "shap_values_summary.json")
                    shap_summary = {
                        "mean_abs_shap": np.abs(results.shap_values).mean(axis=0).tolist(),
                        "feature_names": results.feature_names
                    }
                    with open(shap_path, "w") as f:
                        json.dump(shap_summary, f, indent=2)
                    mlflow.log_artifact(shap_path, "shap")

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