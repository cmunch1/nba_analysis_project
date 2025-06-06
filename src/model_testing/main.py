import sys
import traceback
import logging
from datetime import datetime
import pandas as pd
import numpy as np

from src.common.data_classes import (
    ClassificationMetrics, 
    ModelTrainingResults, 
    PreprocessingResults
)
from .di_container import ModelTestingDIContainer

def main() -> None:
    """Main function to perform model testing with visualization."""
    app_logger = None
    error_handler = None
    try:
        # Initialize container
        container = ModelTestingDIContainer()
        
        # Get core dependencies
        config = container.config()
        app_file_handler = container.app_file_handler()
        app_logger = container.app_logger()

        # initialize the app logger
        log_file = app_file_handler.join_paths(config.log_path, config.model_testing_log_file)
        app_logger.setup(log_file)

        error_handler = container.error_handler()
        data_access = container.data_access()
        data_validator = container.data_validator()


        # Get model testing dependencies
        model_tester = container.model_tester()
        chart_orchestrator = container.chart_orchestrator()
        optimizer = container.optimizer()
        experiment_logger = container.experiment_logger()



        preprocessor = container.preprocessor()


        # Load and validate data
        training_data = data_access.load_dataframe(config.training_data_file)
        validation_data = data_access.load_dataframe(config.validation_data_file)
        
        #data_validator.validate_processed_dataframe(training_data, config.training_data_file)
        #data_validator.validate_processed_dataframe(validation_data, config.validation_data_file)

        app_logger.structured_log(
            logging.INFO,
            "Data loaded and validated",
            training_shape=training_data.shape,
            validation_shape=validation_data.shape
        )

        # Process each enabled model
        results_by_model = {}
        enabled_models = _get_enabled_models(config)

        app_logger.structured_log(
            logging.INFO,
            "Enabled models",
            models=enabled_models
        )

        for model_name in enabled_models:
            try:
                app_logger.structured_log(
                    logging.INFO,
                    f"Processing model: {model_name}",
                    model_name=model_name
                )

                # Process model
                model_results = process_single_model(
                    model_name=model_name,
                    training_dataframe=training_data,
                    validation_dataframe=validation_data,
                    config=config,
                    model_tester=model_tester,
                    data_access=data_access,
                    experiment_logger=experiment_logger,
                    optimizer=optimizer,
                    app_logger=app_logger,
                    chart_orchestrator=chart_orchestrator
                )
                
                results_by_model[model_name] = model_results

            except Exception as e:
                app_logger.structured_log(
                    logging.ERROR,
                    f"Error processing model {model_name}",
                    error=str(e),
                    traceback=traceback.format_exc()
                )
                raise

        # Create comparison visualizations if multiple models
        if len(results_by_model) > 1:
            _create_model_comparison_charts(
                results_by_model,
                chart_orchestrator,
                config,
                app_logger
            )

        app_logger.structured_log(
            logging.INFO,
            "Model testing completed successfully",
            processed_models=list(results_by_model.keys())
        )

    except Exception as e:
        if error_handler and app_logger:
            # Use error handler if available
            raise error_handler.create_error_handler(
                'model_testing',
                "Model testing failed",
                error_message=str(e),
                traceback=traceback.format_exc()
            )
        elif app_logger:
            # If only logger was initialized, use it
            app_logger.structured_log(
                logging.ERROR,
                "Model testing failed",
                error=str(e),
                traceback=traceback.format_exc()
            )
        else:
            # Fallback to basic logging if neither was initialized
            print(f"ERROR: Model testing failed: {str(e)}")
            print(traceback.format_exc())
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
    app_logger,
    chart_orchestrator
) -> ModelTrainingResults:
    """Process a single model with proper visualization integration."""
    
    oof_preprocessing_results = PreprocessingResults()
    val_preprocessing_results = PreprocessingResults()
    
    # Prepare data
    X, y, oof_preprocessing_results, primary_ids_oof = model_tester.prepare_data(
        training_dataframe.copy(), 
        model_name, 
        is_training=True, 
        preprocessing_results=oof_preprocessing_results
    )
    
    X_val, y_val, val_preprocessing_results, primary_ids_val = model_tester.prepare_data(
        validation_dataframe.copy(), 
        model_name, 
        is_training=False, 
        preprocessing_results=val_preprocessing_results
    )

    # Optimize hyperparameters if configured
    if model_tester.get_model_config_value(model_name, 'perform_hyperparameter_optimization', False):
        model_params = optimizer.optimize(
            model_type=model_name,
            X=X,
            y=y
        )
    else:
        model_params = model_tester.get_model_params(model_name)    

    app_logger.structured_log(
        logging.INFO,
        "Model parameters",
        model_name=model_name,
        model_params=model_params
    )

    # Initialize results objects
    oof_results = ModelTrainingResults(X.shape)
    val_results = ModelTrainingResults(X_val.shape)

    # Perform cross-validation if configured
    if config.perform_oof_cross_validation:
        oof_results = process_model_evaluation(
            model_tester=model_tester,
            data_access=data_access,
            experiment_logger=experiment_logger,
            chart_orchestrator=chart_orchestrator,
            app_logger=app_logger,
            model_name=model_name,
            model_params=model_params,
            X=X,
            y=y,
            primary_ids=primary_ids_oof,
            config=config,
            preprocessing_results=oof_preprocessing_results,
            training_results=oof_results,
            experiment_name=config.experiment_name,
            experiment_description=config.experiment_description,
            is_oof=True
        )
    
    # Perform validation set testing if configured
    if config.perform_validation_set_testing:
        val_results = process_model_evaluation(
            model_tester=model_tester,
            data_access=data_access,
            experiment_logger=experiment_logger,
            chart_orchestrator=chart_orchestrator,
            app_logger=app_logger,
            model_name=model_name,
            model_params=model_params,
            X=X,
            y=y,
            X_eval=X_val,
            y_eval=y_val,
            primary_ids=primary_ids_val,
            config=config,
            preprocessing_results=val_preprocessing_results,
            training_results=val_results,
            experiment_name=config.experiment_name,
            experiment_description=config.experiment_description,
            is_oof=False
        )

    return val_results if config.perform_validation_set_testing else oof_results

def process_model_evaluation(
    model_tester,
    data_access,
    experiment_logger,
    chart_orchestrator,
    app_logger,
    model_name: str,
    model_params: dict,
    X,
    y,
    primary_ids,
    config,
    preprocessing_results,
    training_results,
    experiment_name: str,
    experiment_description: str,
    is_oof: bool,
    X_eval=None,
    y_eval=None,
) -> ModelTrainingResults:
    """Process model evaluation with visualization integration."""
    
    eval_type = "OOF" if is_oof else "validation"
    app_logger.structured_log(
        logging.INFO,
        f"Starting {eval_type} evaluation",
        model_name=model_name,
        input_shape=X.shape
    )

    try:
        # Perform evaluation
        if is_oof:
            results = model_tester.perform_oof_cross_validation(
                X, y, model_name, model_params, training_results
            )
        else:
            results = model_tester.perform_validation_set_testing(
                X, y, X_eval, y_eval, model_name, model_params, training_results
            )

        # Update results metadata
        results.is_validation = not is_oof
        results.evaluation_type = eval_type.lower()
        results.model_name = model_name
        results.model_params.update({'hyperparameters': model_params})

        if is_oof:
            results.model_params.update({
                "cross_validation_type": config.cross_validation_type,
                "n_splits": config.n_splits
            })

        # Calculate metrics
        metrics = ClassificationMetrics()  # Initialize metrics object
        metrics = model_tester.calculate_classification_evaluation_metrics(
            results.target_data,
            results.predictions,
            metrics
        )
        results.metrics = metrics

        # Update predictions with optimal threshold
        results.update_predictions(results.predictions, metrics.optimal_threshold)

        # Generate and save visualizations
        charts = chart_orchestrator.create_model_evaluation_charts(results)
        if charts:
            output_dir = f"charts/{model_name}/{eval_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            chart_orchestrator.save_charts(charts, output_dir)

        # Handle predictions saving
        if ((is_oof and config.save_oof_predictions) or 
            (not is_oof and config.save_validation_predictions)):
            _save_predictions(
                results, primary_ids, config, data_access, 
                model_name, is_oof, app_logger
            )

        # Log experiment
        if config.log_experiment:
            app_logger.structured_log(
                logging.INFO,
                f"Logging {eval_type} results"
            )
            experiment_logger.log_experiment(results)

        return results

    except Exception as e:
        raise model_tester.error_handler.create_error_handler(
            'model_testing',
            f"Error in {eval_type} evaluation",
            error_message=str(e),
            model_name=model_name,
            input_shape=X.shape
        )

def _get_enabled_models(config) -> list:
    """Get list of enabled models from config."""
    enabled_models = []
    
    # Process each model in the config
    for model_name in vars(config.models):
        enabled = getattr(config.models, model_name)
        
        if isinstance(enabled, bool) and enabled:
            enabled_models.append(model_name)
            
    return enabled_models

def _save_predictions(
    results: ModelTrainingResults,
    primary_ids,
    config,
    data_access,
    model_name: str,
    is_oof: bool,
    app_logger
) -> None:
    """Save model predictions to file."""
    try:
        predictions_df = results.feature_data.copy()
        predictions_df['target'] = results.target_data
        
        if primary_ids is not None:
            predictions_df.insert(0, config.primary_id_column, primary_ids)
        
        pred_type = "oof" if is_oof else "val"
        predictions_df[f'{pred_type}_predictions'] = results.probability_predictions
        
        output_filename = f"{model_name}_{pred_type}_predictions.csv"
        
        app_logger.structured_log(
            logging.INFO,
            f"Saving {pred_type} predictions",
            output_shape=predictions_df.shape,
            filename=output_filename
        )
        
        data_access.save_dataframes(
            [predictions_df],
            [output_filename]
        )
        
    except Exception as e:
        app_logger.structured_log(
            logging.ERROR,
            "Error saving predictions",
            error=str(e),
            model_name=model_name,
            is_oof=is_oof
        )

def _create_model_comparison_charts(
    results_by_model: dict,
    chart_orchestrator,
    config,
    app_logger
) -> None:
    """Create comparison visualizations for multiple models."""
    try:
        comparison_charts = chart_orchestrator.create_model_comparison_charts(
            list(results_by_model.values())
        )
        
        if comparison_charts:
            output_dir = f"charts/model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            chart_orchestrator.save_charts(comparison_charts, output_dir)
            
        app_logger.structured_log(
            logging.INFO,
            "Model comparison charts created",
            chart_count=len(comparison_charts) if comparison_charts else 0
        )
        
    except Exception as e:
        app_logger.structured_log(
            logging.ERROR,
            "Error creating model comparison charts",
            error=str(e)
        )

if __name__ == "__main__":
    main()
