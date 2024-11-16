import sys
import traceback
import logging
import pandas as pd

from ..logging.logging_setup import setup_logging
from ..logging.logging_utils import log_performance, log_context, structured_log

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

            for model_name in config.models:
                
                model_params = model_tester.get_model_params(model_name)
                
                
                y_oof_predictions= pd.Series(dtype='float64')
                y_val_predictions = pd.Series(dtype='float64')
                
                oof_metrics = None  
                validation_metrics = None

                if config.perform_oof_cross_validation: 
                    
                    y_oof_predictions = model_tester.perform_oof_cross_validation(X, y, model_name, model_params)

                    oof_metrics = model_tester.calculate_classification_evaluation_metrics(y, y_oof_predictions)
                    
                    structured_log(logger, logging.INFO, "OOF cross-validation completed",
                                   accuracy=oof_metrics["accuracy"], precision=oof_metrics["precision"],
                                   recall=oof_metrics["recall"], f1=oof_metrics["f1"], auc=oof_metrics["auc"])

                if config.perform_validation_set_testing:
                    
                    y_val_predictions = model_tester.perform_validation_set_testing(X, y, X_val, y_val, model_name, model_params)
 
                    validation_metrics = model_tester.calculate_classification_evaluation_metrics(y_val, y_val_predictions)    

                    structured_log(logger, logging.INFO, "Validation set testing completed",
                                   accuracy=validation_metrics["accuracy"], precision=validation_metrics["precision"],
                                   recall=validation_metrics["recall"], f1=validation_metrics["f1"], auc=validation_metrics["auc"])
                    
                eval_data = validation_dataframe.copy()
                eval_data["validation_predictions"] = y_val_predictions
                eval_data["oof_predictions"] = y_oof_predictions
                eval_data["target"] = y_val

                experiment_logger.log_experiment(experiment_name=config.experiment_name,
                                                  experiment_description=config.experiment_description,
                                                  model_name=model_name,
                                                  model=model,
                                                  model_params=model_params,
                                                  oof_metrics=oof_metrics,
                                                  validation_metrics=validation_metrics,
                                                  eval_data=eval_data)
  

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

