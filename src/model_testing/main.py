import sys
import traceback
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from ..logging.logging_setup import setup_logging
from ..logging.logging_utils import log_performance, log_context, structured_log
from ..common.data_classes.data_classes import ClassificationMetrics, ModelTrainingResults, PreprocessingResults

from .di_container import ModelTestingDIContainer
from ..error_handling.custom_exceptions import (
    ConfigurationError,
    DataValidationError,
    DataStorageError,
    ModelTestingError,
)

LOG_FILE = "model_testing.log"

@log_performance
def main() -> None:
    """Main function to perform model testing."""
    try:
        # Initialize container
        container = ModelTestingDIContainer()
        
        # Get dependencies
        config = container.config()
        data_access = container.data_access()
        data_validator = container.data_validator()
        model_tester = container.model_tester()
        experiment_logger = container.experiment_logger()
        optimizer = container.optimizer()
        app_logger = container.app_logger()

        # Load and validate data
        data = data_access.load_data()
        data_validator.validate_data(data)

        # Train and evaluate model
        results = model_tester.train_and_evaluate(data)

        # Log experiment with visualizations
        experiment_logger.log_experiment(results)

        app_logger.structured_log(
            logging.INFO,
            "Model testing completed successfully",
            model_name=results.model_name
        )

            structured_log(logger, logging.INFO, "Model testing completed successfully")

    except (ConfigurationError, DataValidationError, 
            DataStorageError, ModelTestingError) as e:
        _handle_known_error(error_logger, e)
    except Exception as e:
        app_logger.structured_log(
            logging.ERROR,
            "Model testing failed",
            error=str(e),
            traceback=traceback.format_exc()
        )
        sys.exit(1)

def process_single_model(
    model_name: str,
    training_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    config,
    model_tester,
    data_access,
    experiment_logger,
    optimizer,
    logger
):
    """Helper function to process a single model"""
    primary_ids_oof = None
    primary_ids_val = None
    oof_metrics = ClassificationMetrics()
    val_metrics = ClassificationMetrics()
    oof_training_results = ModelTrainingResults(training_dataframe.shape)
    val_training_results = ModelTrainingResults(validation_dataframe.shape)
    oof_preprocessing_results = PreprocessingResults()
    val_preprocessing_results = PreprocessingResults()
    
    # prepare the data for the model
    if config.perform_oof_cross_validation or config.perform_hyperparameter_optimization:
        X, y, oof_preprocessing_results, primary_ids_oof = model_tester.prepare_data(
            training_dataframe.copy(), 
            model_name, 
            is_training=True, 
            preprocessing_results=oof_preprocessing_results
        )
    if config.perform_validation_set_testing:
        X_val, y_val, val_preprocessing_results, primary_ids_val = model_tester.prepare_data(
            validation_dataframe.copy(), 
            model_name, 
            is_training=False, 
            preprocessing_results=val_preprocessing_results
        )

    if config.perform_hyperparameter_optimization:
        best_params = optimizer.optimize(
            model_type=model_name,
            X=X,
            y=y
        )
        model_params = best_params
    else:
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
    primary_ids=None,
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
    Ensures proper data filtering for TimeSeriesSplit.
    """
    structured_log(logger, logging.INFO, "Starting model evaluation",
                  is_oof=is_oof,
                  input_shape=X.shape)
    
    try:
        # Perform evaluation
        if is_oof:
            training_results = model_tester.perform_oof_cross_validation(X, y, model_name, model_params, training_results)
        else:
            training_results = model_tester.perform_validation_set_testing(X, y, X_eval, y_eval, model_name, model_params, training_results)

        # Update results with additional information
        training_results.is_validation = not is_oof
        training_results.evaluation_type = "validation" if not is_oof else "oof"
        training_results.model_name = model_name
        
        # Preserve existing params and add hyperparameters
        if not hasattr(training_results.model_params, 'hyperparameters'):
            training_results.model_params = {
                'hyperparameters': model_params
            }
        else:
            training_results.model_params['hyperparameters'] = model_params

        if is_oof:
            training_results.model_params["cross_validation_type"] = config.cross_validation_type
            training_results.model_params["n_splits"] = config.n_splits
        
        # Calculate and update metrics using filtered data
        metrics = model_tester.calculate_classification_evaluation_metrics(
            training_results.target_data,
            training_results.predictions,
            metrics
        )
        training_results.metrics = metrics
        
        # Update predictions with optimal threshold
        training_results.update_predictions(training_results.predictions, metrics.optimal_threshold)
        
        # Handle predictions saving with proper data alignment
        if ((is_oof and config.save_oof_predictions) or 
            (not is_oof and config.save_validation_predictions)):
            
            # Get processed samples mask from training_results
            if hasattr(training_results, 'processed_samples'):
                processed_mask = training_results.processed_samples
            else:
                # Fallback: create mask based on non-NaN predictions
                processed_mask = ~np.isnan(training_results.predictions)
            
            structured_log(logger, logging.INFO, "Preparing predictions dataframe",
                         total_samples=len(y),
                         processed_samples=np.sum(processed_mask))
            
            predictions_df = training_results.feature_data.copy()
            predictions_df['target'] = training_results.target_data
            
            # Add primary IDs if available, ensuring proper filtering
            if primary_ids is not None:
                if len(primary_ids) != len(predictions_df):
                    primary_ids = primary_ids[processed_mask]
                predictions_df.insert(0, config.primary_id_column, primary_ids)
            
            predictions_df[f'{"oof" if is_oof else "val"}_predictions'] = training_results.probability_predictions
            
            structured_log(logger, logging.INFO, f"Saving {'OOF' if is_oof else 'validation'} predictions",
                         output_shape=predictions_df.shape)
            data_access.save_dataframes(
                [predictions_df], 
                [f"{model_name}_{'oof' if is_oof else 'val'}_predictions.csv"]
            )

        # Add preprocessing results
        training_results.preprocessing_results = preprocessing_results
        
        # Log experiment if configured
        if config.log_experiment:
            structured_log(logger, logging.INFO, f"Logging {'OOF' if is_oof else 'validation'} results")
            experiment_logger.log_experiment(training_results)
        
        structured_log(logger, logging.INFO, "Model evaluation completed",
                      final_shape=training_results.feature_data.shape,
                      predictions_shape=training_results.predictions.shape)
        
        return training_results

    except Exception as e:
        raise ModelTestingError("Error in model evaluation",
                              error_message=str(e),
                              model_name=model_name,
                              is_oof=is_oof)


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

