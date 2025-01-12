import sys
import traceback
import logging
import pandas as pd
from datetime import datetime

from ..logging.logging_setup import setup_logging
from ..logging.logging_utils import log_performance, log_context, structured_log
from .data_classes import ClassificationMetrics, ModelTrainingResults, PreprocessingResults


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
            
            for model_name in config.models:

                primary_ids_oof = None
                primary_ids_val = None
                oof_metrics = ClassificationMetrics()
                val_metrics = ClassificationMetrics()
                oof_training_results = ModelTrainingResults(training_dataframe.shape)
                val_training_results = ModelTrainingResults(validation_dataframe.shape)
                oof_preprocessing_results = PreprocessingResults()
                val_preprocessing_results = PreprocessingResults()
                
                # prepare the data for the model, including preprocessing steps saved in the results dataclass
                X, y, oof_preprocessing_results, primary_ids_oof = model_tester.prepare_data(training_dataframe.copy(), 
                                                                            model_name, 
                                                                            is_training=True, 
                                                                            preprocessing_results=oof_preprocessing_results)
                X_val, y_val, val_preprocessing_results, primary_ids_val = model_tester.prepare_data(validation_dataframe.copy(), 
                                                                                    model_name, 
                                                                                    is_training=False, 
                                                                                    preprocessing_results=val_preprocessing_results)
                
                model_params = model_tester.get_model_params(model_name)
                
                if config.perform_oof_cross_validation:
                    oof_training_results = process_model_evaluation(
                        model_tester, data_access, experiment_logger, logger,
                        model_name, model_params, X, y,
                        primary_ids=primary_ids_oof,
                        config=config,
                        preprocessing_results=oof_preprocessing_results,
                        training_results=oof_training_results,
                        metrics=oof_metrics,
                        experiment_name=config.experiment_name,
                        experiment_description=config.experiment_description,
                        is_oof=True
                    )
                
                if config.perform_validation_set_testing:
                    val_training_results = process_model_evaluation(
                        model_tester, data_access, experiment_logger, logger,
                        model_name, model_params, X, y,
                        X_eval=X_val, 
                        y_eval=y_val,
                        primary_ids=primary_ids_val,
                        config=config,
                        preprocessing_results=val_preprocessing_results,
                        training_results=val_training_results,
                        metrics=val_metrics,
                        experiment_name=config.experiment_name,
                        experiment_description=config.experiment_description,
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
    primary_ids=None,  # Add this parameter
    config=None,
    preprocessing_results=None,
    training_results=None,
    metrics=None,
    experiment_name: str = None,
    experiment_description: str = None,
    is_oof: bool = True
) -> tuple[ModelTrainingResults, ClassificationMetrics]:
    """
    Unified function to handle both OOF cross-validation and validation set testing.
    
    Args:
        is_oof: If True, performs OOF cross-validation. If False, performs validation testing.
    """
    # Perform evaluation
    if is_oof:
        training_results = model_tester.perform_oof_cross_validation(X, y, model_name, model_params, training_results)
        eval_data = X
        eval_y = y
    else:
        training_results = model_tester.perform_validation_set_testing(X, y, X_eval, y_eval, model_name, model_params, training_results)
        eval_data = X_eval
        eval_y = y_eval

    # Update results with additional information
    training_results.is_validation = not is_oof
    training_results.evaluation_type = "validation" if not is_oof else "oof"
    training_results.model_name = model_name
    training_results.model_params = model_params
    if is_oof:
        training_results.model_params["cross_validation_type"]= config.cross_validation_type
        training_results.model_params["n_splits"]= config.n_splits
    
    # Update feature data
    training_results.update_feature_data(eval_data, eval_y)
    
    # Calculate and update metrics
    metrics = model_tester.calculate_classification_evaluation_metrics(eval_y, training_results.predictions, metrics)
    training_results.metrics = metrics
    
    # Update predictions with optimal threshold from metrics
    training_results.update_predictions(training_results.predictions, metrics.optimal_threshold)
    
    # Handle predictions saving
    if ((is_oof and config.save_oof_predictions) or 
        (not is_oof and config.save_validation_predictions)):
        
        predictions_df = training_results.feature_data.copy()
        predictions_df['target'] = training_results.target_data
        if primary_ids is not None:
            predictions_df.insert(0, config.primary_id_column, primary_ids)
        predictions_df[f'{"oof" if is_oof else "val"}_predictions'] = training_results.probability_predictions

        
        structured_log(logger, logging.INFO, f"Saving {'OOF' if is_oof else 'validation'} predictions")
        data_access.save_dataframes(
            [predictions_df], 
            [f"{model_name}_{'oof' if is_oof else 'val'}_predictions.csv"]
        )

    # Add preprocessing results to the results dataclass
    training_results.preprocessing_results = preprocessing_results
    
    # Log experiment if configured
    if config.log_experiment:
        structured_log(logger, logging.INFO, f"Logging {'OOF' if is_oof else 'validation'} results")
        experiment_logger.log_experiment(training_results)
    
    return training_results


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

