"""Pruning workflow for baseline vs pruned model comparison."""

import logging
import pandas as pd
from typing import Tuple

from ml_framework.framework.data_classes import ModelTrainingResults


def run_with_pruning(
    model_name: str,
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    config,
    model_tester,
    data_access,
    experiment_logger,
    optimizer,
    app_logger,
    chart_orchestrator,
    feature_auditor,
    feature_pruner,
    pruning_comparison,
    process_single_model_func
) -> Tuple[ModelTrainingResults, pd.DataFrame]:
    """
    Execute baseline + pruning workflow.

    Args:
        model_name: Name of the model
        training_data: Training DataFrame
        validation_data: Validation DataFrame
        config: Configuration manager
        model_tester: Model tester instance
        data_access: Data access instance
        experiment_logger: Experiment logger instance
        optimizer: Hyperparameter optimizer instance
        app_logger: Application logger
        chart_orchestrator: Chart orchestrator instance
        feature_auditor: Feature auditor instance
        feature_pruner: Feature pruner instance
        pruning_comparison: Pruning comparison instance
        process_single_model_func: Function to process a single model

    Returns:
        Tuple of (final_results, comparison_report)
    """
    pruning_config = config.core.model_testing_config.feature_pruning

    app_logger.structured_log(
        logging.INFO,
        "Starting pruning workflow",
        model_name=model_name,
        pruning_enabled=True
    )

    # ============================================================
    # Run 1: Baseline (All Features)
    # ============================================================
    app_logger.structured_log(
        logging.INFO,
        "Running baseline model with all features",
        model_name=model_name,
        n_features=training_data.shape[1]
    )

    baseline_results = process_single_model_func(
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

    # ============================================================
    # Identify Drop Candidates
    # ============================================================
    if baseline_results.feature_audit is None:
        app_logger.structured_log(
            logging.WARNING,
            "No feature audit available, cannot prune features",
            model_name=model_name
        )
        return baseline_results, None

    drop_candidates = feature_pruner.identify_drop_candidates(
        baseline_results.feature_audit,
        threshold=pruning_config.drop_candidate_threshold
    )

    if len(drop_candidates) == 0:
        app_logger.structured_log(
            logging.INFO,
            "No features flagged for dropping, skipping pruning",
            model_name=model_name
        )
        return baseline_results, None

    # ============================================================
    # Validate Pruning Safety
    # ============================================================
    is_safe, safety_message = feature_pruner.validate_pruning_safety(
        baseline_results.feature_names,
        drop_candidates
    )

    if not is_safe:
        app_logger.structured_log(
            logging.WARNING,
            "Pruning validation failed, skipping pruning",
            model_name=model_name,
            reason=safety_message
        )
        return baseline_results, None

    # ============================================================
    # Prune Datasets
    # ============================================================
    app_logger.structured_log(
        logging.INFO,
        "Pruning datasets",
        n_features_to_drop=len(drop_candidates),
        pct_to_drop=f"{len(drop_candidates)/len(baseline_results.feature_names)*100:.1f}%"
    )

    training_data_pruned = feature_pruner.prune_dataset(training_data, drop_candidates)
    validation_data_pruned = feature_pruner.prune_dataset(validation_data, drop_candidates)

    # ============================================================
    # Run 2: Pruned Model
    # ============================================================
    app_logger.structured_log(
        logging.INFO,
        "Running pruned model",
        model_name=model_name,
        n_features=training_data_pruned.shape[1]
    )

    pruned_results = process_single_model_func(
        model_name=model_name,
        training_dataframe=training_data_pruned,
        validation_dataframe=validation_data_pruned,
        config=config,
        model_tester=model_tester,
        data_access=data_access,
        experiment_logger=experiment_logger,
        optimizer=optimizer,
        app_logger=app_logger,
        chart_orchestrator=chart_orchestrator,
        feature_auditor=feature_auditor
    )

    # ============================================================
    # Compare Results
    # ============================================================
    app_logger.structured_log(
        logging.INFO,
        "Generating pruning comparison report",
        model_name=model_name
    )

    comparison_df = pruning_comparison.compare_runs(
        baseline_results,
        pruned_results,
        drop_candidates
    )

    # Save comparison report if configured
    if pruning_config.save_comparison_report:
        pruning_summary = feature_pruner.get_pruning_summary(
            baseline_results.feature_names,
            drop_candidates,
            baseline_results.feature_audit
        )

        eval_type = "oof" if baseline_results.n_folds > 0 else "validation"
        pruning_comparison.save_comparison_report(
            comparison_df,
            model_name,
            eval_type,
            drop_candidates,
            pruning_summary
        )

    # ============================================================
    # Decide Which Model to Keep
    # ============================================================
    recommendation = comparison_df[comparison_df['metric'] == 'recommendation']['baseline'].values[0]

    if recommendation == "ACCEPT_PRUNED":
        app_logger.structured_log(
            logging.INFO,
            "Pruning successful, using pruned model",
            model_name=model_name,
            recommendation=recommendation
        )
        final_results = pruned_results
    elif recommendation == "REJECT_PRUNED":
        app_logger.structured_log(
            logging.WARNING,
            "Pruning degraded performance too much, reverting to baseline",
            model_name=model_name,
            recommendation=recommendation
        )
        final_results = baseline_results
    else:  # CONSIDER_PRUNED or MANUAL_REVIEW
        app_logger.structured_log(
            logging.INFO,
            "Pruning results inconclusive, defaulting to pruned model",
            model_name=model_name,
            recommendation=recommendation
        )
        final_results = pruned_results

    return final_results, comparison_df
