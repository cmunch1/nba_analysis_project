import sys
import traceback
import logging
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
import shap
import matplotlib.pyplot as plt


from ..logging.logging_setup import setup_logging
from ..logging.logging_utils import log_performance, log_context, structured_log

from .di_container import DIContainer
from ..error_handling.custom_exceptions import (
    ConfigurationError,
    DataValidationError,
    DataStorageError,
    ModelTrainingError,
)

LOG_FILE = "model_training.log"

@log_performance
def main() -> None:
    """
    Main function to perform model training on engineered NBA data.
    """

    container = DIContainer()
    config = container.config()
    data_access = container.data_access()
    data_validator = container.data_validator()
    model_trainer = container.model_trainer()

    try:
        error_logger = setup_logging(config, LOG_FILE)
        logger = logging.getLogger(__name__)

        structured_log(logger, logging.INFO, "Starting model training", 
                       app_version=config.app_version, 
                       environment=config.environment,
                       log_level=config.log_level,
                       config_summary=str(config.__dict__))

        with log_context(app_version=config.app_version, environment=config.environment):
            
            training_dataframe = data_access.load_dataframe(config.training_data_file)
            validation_dataframe = data_access.load_dataframe(config.validation_data_file)
          
            X, y = model_trainer.prepare_data(training_dataframe)
            X_val, y_val = model_trainer.prepare_data(validation_dataframe)

             
            for model_name in config.models:
                
                model, model_params = model_trainer.get_model_params(model_name)
                
                for cv_type in config.cross_validation_types:
                    metrics = {}
                     
                    # Perform cross-validation
                    metrics = model_trainer.perform_cross_validation(X, y, model_name, model, cv_type, config.n_splits)

                    # run model on validation data
                    y_val_pred = model.predict(X=X_val)

                    eval_data = X_val
                    eval_data["target"] = y_val
                    eval_data["predictions"] = y_val_pred
                               
                    with mlflow.start_run():
                        mlflow.log_params(model_params)

                        # Log cross-validation scores
                        mlflow.log_metric("oof_accuracy", metrics["overall_accuracy"])
                        mlflow.log_metric("oof_auc", metrics["overall_auc"])

                        # Log model
                        mlflow.sklearn.log_model(model, model_name)

                        # Create the PandasDataset for use in mlflow evaluate
                        pd_dataset = mlflow.data.from_pandas(
                            eval_data, predictions="predictions", targets="target"
    )
                        mlflow.log_input(pd_dataset, context="validation")

                        result = mlflow.evaluate(
                            data=eval_data,
                            targets="target",
                            predictions="predictions",
                            model_type="classifier",
                        )

                        print(f"metrics:\n{result.metrics}")
                        print(f"artifacts:\n{result.artifacts}")

            structured_log(logger, logging.INFO, "Model training completed successfully", 
                           oof_accuracy=metrics["overall_accuracy"],
                           oof_auc=metrics["overall_auc"],)

    except (ConfigurationError, DataValidationError, 
            DataStorageError, ModelTrainingError) as e:
        _handle_known_error(error_logger, e)
    except Exception as e:
        _handle_unexpected_error(error_logger, e)

def _handle_known_error(error_logger, e):
    structured_log(error_logger, logging.ERROR, f"{type(e).__name__} occurred", 
                   error_message=str(e),
                   error_type=type(e).__name__,
                   traceback=traceback.format_exc())
    sys.exit(1)

def _handle_unexpected_error(error_logger, e):
    structured_log(error_logger, logging.CRITICAL, "Unexpected error occurred", 
                   error_message=str(e),
                   error_type=type(e).__name__,
                   traceback=traceback.format_exc())
    sys.exit(6)

if __name__ == "__main__":
    main()