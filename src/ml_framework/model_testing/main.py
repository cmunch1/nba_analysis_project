import sys
import traceback
import logging
from datetime import datetime
import pandas as pd


from ml_framework.framework.data_classes import (
    ClassificationMetrics,
    ModelTrainingResults,
    PreprocessingResults
)
from .di_container import ModelTestingDIContainer
from .pruning_workflow import run_with_pruning


def main() -> None:
    """Main function to perform model testing with visualization."""
    app_logger = None
    error_handler = None
    try:
        # Initialize container
        container = ModelTestingDIContainer()

        # Get core dependencies
        config = container.config()
        app_logger = container.app_logger()

        # Initialize the app logger
        app_logger.setup(config.core.app_logging_config.model_testing_log_file)

        error_handler = container.error_handler()
        data_access = container.data_access()


        # Get model testing dependencies
        model_tester = container.model_tester()
        chart_orchestrator = container.chart_orchestrator()
        optimizer = container.optimizer()
        experiment_logger = container.experiment_logger()
        feature_auditor = container.feature_auditor()
        feature_pruner = container.feature_pruner()
        pruning_comparison = container.pruning_comparison()

        # Load and validate data
        training_data = data_access.load_dataframe(config.training_data_file)
        validation_data = data_access.load_dataframe(config.validation_data_file)
        

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

                # Check if pruning is enabled
                if config.core.model_testing_config.enable_feature_pruning:
                    # Run with pruning workflow (baseline + pruned)
                    model_results, comparison_df = run_with_pruning(
                        model_name=model_name,
                        training_data=training_data,
                        validation_data=validation_data,
                        config=config,
                        model_tester=model_tester,
                        data_access=data_access,
                        experiment_logger=experiment_logger,
                        optimizer=optimizer,
                        app_logger=app_logger,
                        chart_orchestrator=chart_orchestrator,
                        feature_auditor=feature_auditor,
                        feature_pruner=feature_pruner,
                        pruning_comparison=pruning_comparison,
                        process_single_model_func=process_single_model
                    )
                else:
                    # Standard single-run workflow
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
                        chart_orchestrator=chart_orchestrator,
                        feature_auditor=feature_auditor
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
    chart_orchestrator,
    feature_auditor
) -> ModelTrainingResults:
    """Process a single model with proper visualization integration."""

    # Create local alias for model testing config (used frequently in this function)
    model_cfg = config.core.model_testing_config

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
    should_optimize = model_tester.get_model_config_value(model_name, 'perform_hyperparameter_optimization', False)
    app_logger.structured_log(
        logging.INFO,
        "Checking hyperparameter optimization setting",
        model_name=model_name,
        should_optimize=should_optimize
    )

    if should_optimize:
        app_logger.structured_log(
            logging.INFO,
            "Starting hyperparameter optimization",
            model_name=model_name
        )
        model_params = optimizer.optimize(
            model_type=model_name,
            X=X,
            y=y
        )
    else:
        app_logger.structured_log(
            logging.INFO,
            "Skipping hyperparameter optimization, using existing params",
            model_name=model_name
        )
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
    if model_cfg.perform_oof_cross_validation:
        oof_results = process_model_evaluation(
            model_tester=model_tester,
            data_access=data_access,
            experiment_logger=experiment_logger,
            chart_orchestrator=chart_orchestrator,
            feature_auditor=feature_auditor,
            app_logger=app_logger,
            model_name=model_name,
            model_params=model_params,
            X=X,
            y=y,
            primary_ids=primary_ids_oof,
            config=config,
            preprocessing_results=oof_preprocessing_results,
            training_results=oof_results,
            experiment_name=model_cfg.experiment_name,
            experiment_description=model_cfg.experiment_description,
            is_oof=True
        )

    # Perform validation set testing if configured
    if model_cfg.perform_validation_set_testing:
        val_results = process_model_evaluation(
            model_tester=model_tester,
            data_access=data_access,
            experiment_logger=experiment_logger,
            chart_orchestrator=chart_orchestrator,
            feature_auditor=feature_auditor,
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
            experiment_name=model_cfg.experiment_name,
            experiment_description=model_cfg.experiment_description,
            is_oof=False
        )

    return val_results if model_cfg.perform_validation_set_testing else oof_results

def process_model_evaluation(
    model_tester,
    data_access,
    experiment_logger,
    chart_orchestrator,
    feature_auditor,
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

    # Create local alias for model testing config
    model_cfg = config.core.model_testing_config

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
                "cross_validation_type": model_cfg.cross_validation_type,
                "n_splits": model_cfg.n_splits
            })

        # Calculate metrics on uncalibrated predictions
        metrics = ClassificationMetrics()  # Initialize metrics object
        metrics = model_tester.calculate_classification_evaluation_metrics(
            results.target_data,
            results.predictions,
            metrics
        )
        results.metrics = metrics

        # Update predictions with optimal threshold
        results.update_predictions(results.predictions, metrics.optimal_threshold)

        # Apply probability calibration if enabled
        # Use results.target_data which contains the actual targets for predictions
        # (important for OOF where predictions may be subset of full training data)
        results = model_tester.calibrate_probabilities(
            results=results,
            X_cal=None,  # Not needed for direct probability calibration
            y_cal=results.target_data,
            model_name=model_name
        )

        # If calibration was applied, calculate calibrated metrics
        if results.calibrated_predictions is not None:
            calibrated_metrics = ClassificationMetrics()
            calibrated_metrics = model_tester.calculate_classification_evaluation_metrics(
                results.target_data,
                results.calibrated_predictions,
                calibrated_metrics
            )

            # Log comparison between uncalibrated and calibrated
            app_logger.structured_log(
                logging.INFO,
                "Calibration metrics comparison",
                model_name=model_name,
                uncalibrated_auc=float(metrics.auc),
                calibrated_auc=float(calibrated_metrics.auc),
                brier_improvement=results.calibration_metrics.get('brier_score_improvement') if results.calibration_metrics else None,
                ece_improvement=results.calibration_metrics.get('ece_improvement') if results.calibration_metrics else None
            )

        # Generate and save visualizations
        charts = chart_orchestrator.create_model_evaluation_charts(results)
        if charts:
            # Use configured charts output directory
            base_charts_dir = getattr(model_cfg, 'charts_output_dir', 'src/ml_framework/ml_artifacts/visualizations/charts')
            output_dir = f"{base_charts_dir}/{model_name}/{eval_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            chart_orchestrator.save_charts(charts, output_dir)

        # Generate feature audit if configured
        if model_cfg.generate_feature_audit:
            try:
                app_logger.structured_log(
                    logging.INFO,
                    f"Generating feature audit for {eval_type}",
                    model_name=model_name
                )

                # Use evaluation data for permutation importance
                audit_X = X_eval if X_eval is not None else X
                audit_y = y_eval if y_eval is not None else y

                # Create audit
                feature_audit = feature_auditor.create_audit(
                    results=results,
                    X_eval=audit_X,
                    y_eval=audit_y,
                    run_id=None,  # Will be populated by experiment logger if available
                    experiment_id=None
                )

                # Save audit if configured
                if model_cfg.save_feature_audit:
                    audit_path = feature_auditor.save_audit(
                        audit_df=feature_audit,
                        model_name=model_name,
                        eval_type=eval_type.lower(),
                        run_id=None
                    )
                    app_logger.structured_log(
                        logging.INFO,
                        f"Feature audit saved",
                        audit_path=audit_path
                    )

                # Store audit in results for experiment logging
                results.feature_audit = feature_audit

            except Exception as e:
                app_logger.structured_log(
                    logging.ERROR,
                    "Feature audit generation failed",
                    error=str(e),
                    model_name=model_name
                )
                # Don't fail the entire evaluation if audit fails

        # Handle predictions saving
        if ((is_oof and model_cfg.save_oof_predictions) or
            (not is_oof and model_cfg.save_validation_predictions)):
            _save_predictions(
                results, primary_ids, config, data_access,
                model_name, is_oof, app_logger
            )

        # Log experiment
        if model_cfg.log_experiment:
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
    for model_name in vars(config.core.model_testing_config.models):
        enabled = getattr(config.core.model_testing_config.models, model_name)

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
    """Save model predictions to file, including calibrated predictions if available."""
    try:
        predictions_df = results.feature_data.copy()
        predictions_df['target'] = results.target_data

        if primary_ids is not None:
            predictions_df.insert(0, config.core.model_testing_config.primary_id_column, primary_ids)

        pred_type = "oof" if is_oof else "val"

        # Save uncalibrated predictions
        predictions_df[f'{pred_type}_predictions_uncalibrated'] = results.probability_predictions

        # Save calibrated predictions if available
        if results.calibrated_predictions is not None:
            predictions_df[f'{pred_type}_predictions_calibrated'] = results.calibrated_predictions

            # Add comparison columns for easy analysis
            import numpy as np
            predictions_df[f'{pred_type}_calibration_adjustment'] = (
                results.calibrated_predictions - results.probability_predictions
            )
            predictions_df[f'{pred_type}_abs_adjustment'] = np.abs(
                results.calibrated_predictions - results.probability_predictions
            )

            app_logger.structured_log(
                logging.INFO,
                f"Including calibrated predictions in save",
                mean_adjustment=float(predictions_df[f'{pred_type}_calibration_adjustment'].mean()),
                max_abs_adjustment=float(predictions_df[f'{pred_type}_abs_adjustment'].max())
            )

        output_filename = f"{model_name}_{pred_type}_predictions.csv"

        app_logger.structured_log(
            logging.INFO,
            f"Saving {pred_type} predictions",
            output_shape=predictions_df.shape,
            filename=output_filename,
            has_calibrated=results.calibrated_predictions is not None
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
            # Use configured charts output directory
            base_charts_dir = getattr(config.core.model_testing_config, 'charts_output_dir', 'src/ml_framework/ml_artifacts/visualizations/charts')
            output_dir = f"{base_charts_dir}/model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
