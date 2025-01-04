import sys
import traceback
import logging
import pandas as pd
from datetime import datetime

from ..logging.logging_setup import setup_logging
from ..logging.logging_utils import log_performance, log_context, structured_log
from .data_classes import ClassificationMetrics, ModelTrainingResults


from .di_container import DIContainer
from ..error_handling.custom_exceptions import (
    ConfigurationError,
    DataValidationError,
    DataStorageError,
    ModelTestingError,
)

LOG_FILE = "model_testing.log"

@log_performance
def main() -> None:
    """
    Main function to perform model testing on engineered data.
    """

    container = DIContainer()
    config = container.config()
    data_access = container.data_access()
    data_validator = container.data_validator()
    model_tester = container.model_tester()
    experiment_logger = container.experiment_logger()
    
    try:
        error_logger = setup_logging(config, LOG_FILE)
        logger = logging.getLogger(__name__)

        structured_log(logger, logging.INFO, "Starting model testing", 
                       app_version=config.app_version, 
                       environment=config.environment,
                       log_level=config.log_level,
                       config_summary=str(config.__dict__))

        with log_context(app_version=config.app_version, environment=config.environment):

            training_dataframe = data_access.load_dataframe(config.training_data_file)
            validation_dataframe = data_access.load_dataframe(config.validation_data_file)
            
            X, y = model_tester.prepare_data(training_dataframe.copy())
            X_val, y_val = model_tester.prepare_data(validation_dataframe.copy())

            oof_metrics = ClassificationMetrics()
            val_metrics = ClassificationMetrics()
            oof_training_results = ModelTrainingResults(X.shape)
            val_training_results = ModelTrainingResults(X_val.shape)

            if config.experiment_name == "Default":
                experiment_name = "Experiment_" + datetime.now().strftime("%Y%m%d%H%M%S")
            else:
                experiment_name=config.experiment_name

            if config.experiment_description == "Default":
                experiment_description = "Experiment_" + datetime.now().strftime("%Y%m%d%H%M%S")
            else:
                experiment_description=config.experiment_description
   

            for model_name in config.models:
                model_params = model_tester.get_model_params(model_name)
                
                if config.perform_oof_cross_validation:
                    oof_training_results, oof_metrics = process_model_evaluation(
                        model_tester, data_access, experiment_logger, logger,
                        model_name, model_params, X, y,
                        config=config, experiment_name=experiment_name,
                        experiment_description=experiment_description,
                        is_oof=True
                    )
                
                if config.perform_validation_set_testing:
                    val_training_results, val_metrics = process_model_evaluation(
                        model_tester, data_access, experiment_logger, logger,
                        model_name, model_params, X, y, X_val, y_val,
                        config=config, experiment_name=experiment_name,
                        experiment_description=experiment_description,
                        is_oof=False
                    )
            structured_log(logger, logging.INFO, "Model testing completed successfully")

    except (ConfigurationError, DataValidationError, 
            DataStorageError, ModelTestingError) as e:
        _handle_known_error(error_logger, e)
    except Exception as e:
        _handle_unexpected_error(error_logger, e)


def process_model_evaluation(
    model_tester,
    data_access,
    experiment_logger,
    logger,
    model_name: str,
    model_params: dict,
    X,
    y,
    X_eval=None,  # None for OOF, validation data for val testing
    y_eval=None,
    config=None,
    experiment_name: str = None,
    experiment_description: str = None,
    is_oof: bool = True
) -> tuple[ModelTrainingResults, ClassificationMetrics]:
    """
    Unified function to handle both OOF cross-validation and validation set testing.
    
    Args:
        is_oof: If True, performs OOF cross-validation. If False, performs validation testing.
    """
    prefix = "oof" if is_oof else "val"
    
    # Perform evaluation
    if is_oof:
        results = model_tester.perform_oof_cross_validation(X, y, model_name, model_params)
        eval_y = y
    else:
        results = model_tester.perform_validation_set_testing(X, y, X_eval, y_eval, 
                                                            model_name, model_params)
        eval_y = y_eval
        
    # Calculate metrics
    metrics = model_tester.calculate_classification_evaluation_metrics(eval_y, results.predictions)
    
    # Log completion
    structured_log(logger, logging.INFO, 
                  f"{'OOF cross-validation' if is_oof else 'Validation set testing'} completed",
                  accuracy=metrics.accuracy, precision=metrics.precision,
                  recall=metrics.recall, f1=metrics.f1, auc=metrics.auc)
    
    # Handle predictions saving and experiment logging if needed
    if config.save_oof_predictions or config.save_validation_predictions or config.log_experiment:
        eval_data = (X if is_oof else X_eval).copy()
        eval_data[f"{prefix}_predictions"] = results.predictions
        eval_data["target"] = eval_y.copy()
        
        if (is_oof and config.save_oof_predictions) or (not is_oof and config.save_validation_predictions):
            structured_log(logger, logging.INFO, f"Saving {prefix} predictions")
            data_access.save_dataframes([eval_data], [f"{experiment_name}_{prefix}_predictions.csv"])
            
        if config.log_experiment:
            structured_log(logger, logging.INFO, f"Logging {prefix} predictions")
            experiment_logger.log_experiment(
                experiment_name=experiment_name,
                experiment_description=experiment_description,
                model_name=model_name,
                model=results.model,
                model_params=model_params,
                **{f"{prefix}_metrics": metrics, f"{prefix}_data": eval_data}
            )
    
    return results, metrics



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

