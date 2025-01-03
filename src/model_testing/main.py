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

            oof_metrics = ClassificationMetrics()
            val_metrics = ClassificationMetrics()
            oof_training_results = ModelTrainingResults()
            val_training_results = ModelTrainingResults()

            training_dataframe = data_access.load_dataframe(config.training_data_file)
            validation_dataframe = data_access.load_dataframe(config.validation_data_file)
            
            X, y = model_tester.prepare_data(training_dataframe.copy())
            X_val, y_val = model_tester.prepare_data(validation_dataframe.copy())

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
                    
                    oof_training_results = model_tester.perform_oof_cross_validation(X, y, model_name, model_params)
                    oof_metrics = model_tester.calculate_classification_evaluation_metrics(y, oof_training_results.predictions)

                    if config.save_oof_predictions or config.log_experiment:
                        oof_data = X.copy()
                        oof_data["oof_predictions"] = oof_training_results.predictions
                        oof_data["target"] = y.copy() 

                    structured_log(logger, logging.INFO, "OOF cross-validation completed",
                                   accuracy=oof_metrics.accuracy, precision=oof_metrics.precision,
                                   recall=oof_metrics.recall, f1=oof_metrics.f1, auc=oof_metrics.auc)
                                       
                    if config.save_oof_predictions:
                        structured_log(logger, logging.INFO, "Saving OOF predictions",
                                       )
                        data_access.save_dataframes([oof_data], [f"{experiment_name}_oof_predictions.csv"])

                    if config.log_experiment:
                        structured_log(logger, logging.INFO, "Logging OOF predictions",
                                       )
                        
                        experiment_logger.log_experiment(experiment_name=experiment_name,
                                                  experiment_description=experiment_description,
                                                  model_name=model_name,
                                                  model=oof_training_results.model,
                                                  model_params=model_params,
                                                  oof_metrics=oof_metrics,
                                                  oof_data=oof_data)

                
                if config.perform_validation_set_testing:
                    
                    val_training_results = model_tester.perform_validation_set_testing(X, y, X_val, y_val, model_name, model_params)
 
                    val_metrics = model_tester.calculate_classification_evaluation_metrics(y_val, val_training_results.predictions) 

                    if config.save_validation_predictions or config.log_experiment:
                        val_data = X_val.copy()
                        val_data["val_predictions"] = val_training_results.predictions
                        val_data["target"] = y_val.copy()      

                    structured_log(logger, logging.INFO, "Validation set testing completed",
                                   accuracy=val_metrics.accuracy, precision=val_metrics.precision,
                                   recall=val_metrics.recall, f1=val_metrics.f1, auc=val_metrics.auc)

                    
                    if config.save_validation_predictions:
                        
                        structured_log(logger, logging.INFO, "Saving validation predictions",
                                       )
                        data_access.save_dataframes([val_data], [f"{experiment_name}_validation_predictions.csv"])
                    
                    if config.log_experiment:
                        structured_log(logger, logging.INFO, "Logging validation predictions",
                                       )    
                        experiment_logger.log_experiment(experiment_name=experiment_name,
                                                  experiment_description=experiment_description,
                                                  model_name=model_name,
                                                  model=val_training_results.model,
                                                  model_params=model_params,
                                                  val_metrics=val_metrics,
                                                  val_data=val_data)

  

            structured_log(logger, logging.INFO, "Model testing completed successfully")

    except (ConfigurationError, DataValidationError, 
            DataStorageError, ModelTestingError) as e:
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

