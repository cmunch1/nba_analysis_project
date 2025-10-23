"""Simplified NBA Game Prediction Pipeline

This module runs daily AFTER feature_engineering to generate predictions.

Simplified workflow (matching your proven approach):
1. Load engineered data (already contains today's games with features)
2. Filter to today's games only (using todays_games_ids.csv)
3. Load production model
4. Make predictions (preprocessing automatic via ModelPredictor)
5. Apply postprocessing (calibration + conformal prediction)
6. Save predictions

No feature engineering here - that's handled by feature_engineering/main.py
"""

import sys
import traceback
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

from ml_framework.core.config_management.config_manager import ConfigManager
from ml_framework.core.app_logging.app_logger import AppLogger
from ml_framework.core.app_file_handling.app_file_handler import LocalAppFileHandler
from ml_framework.core.error_handling.error_handler_factory import ErrorHandlerFactory
from ml_framework.framework.data_access.csv_data_access import CSVDataAccess
from ml_framework.model_registry.mlflow_model_registry import MLflowModelRegistry
from ml_framework.inference.model_predictor import ModelPredictor
from ml_framework.postprocessing.probability_calibrator import ProbabilityCalibrator
from ml_framework.postprocessing.conformal_predictor import ConformalPredictor


LOG_FILE = "inference.log"


def main() -> None:
    """
    Main function to run the NBA game prediction pipeline.
    """
    app_logger = None

    try:
        # Initialize using DI container pattern (like feature_engineering)
        from ml_framework.core.common_di_container import CommonDIContainer

        container = CommonDIContainer()
        container.init_resources()
        container.wire(modules=[__name__])

        # Get dependencies from container
        config = container.config()
        app_logger = container.app_logger()

        # Setup logging FIRST (before creating other components that log during init)
        app_logger.setup(LOG_FILE)

        # Now get remaining dependencies
        app_file_handler = container.app_file_handler()
        error_handler = container.error_handler_factory()
        data_access = container.data_access()
        model_registry = container.model_registry()

        app_logger.structured_log(
            logging.INFO,
            "Starting NBA inference pipeline (simplified)",
            app_version=config.app_version,
            environment=config.environment
        )

        with app_logger.log_context(app_version=config.app_version, environment=config.environment):

            # Step 1: Load today's game IDs
            app_logger.structured_log(logging.INFO, "Loading today's game IDs")
            # Load from newly_scraped directory (not cumulative)
            todays_games_path = Path(config.newly_scraped_directory) / config.todays_games_ids_file
            todays_games_df = pd.read_csv(todays_games_path)

            if todays_games_df.empty:
                app_logger.structured_log(
                    logging.INFO,
                    "No games scheduled for today - nothing to predict"
                )
                return

            todays_game_ids = todays_games_df['game_id'].tolist()
            app_logger.structured_log(
                logging.INFO,
                "Today's games loaded",
                num_games=len(todays_game_ids),
                game_ids=todays_game_ids
            )

            # Step 2: Load engineered data (includes today's games with features)
            app_logger.structured_log(
                logging.INFO,
                "Loading engineered data",
                file_name=config.engineered_data_file
            )
            # Load from engineered directory
            engineered_data_path = Path(config.engineered_data_directory) / config.engineered_data_file
            engineered_df = pd.read_csv(engineered_data_path)

            # Step 3: Filter to only today's games
            todays_features = engineered_df[engineered_df['game_id'].isin(todays_game_ids)]

            if todays_features.empty:
                raise error_handler.create_error_handler(
                    'data_validation',
                    "Today's games not found in engineered data - ensure feature_engineering ran with today's matchups",
                    expected_game_ids=todays_game_ids
                )

            app_logger.structured_log(
                logging.INFO,
                "Today's games filtered from engineered data",
                num_rows=len(todays_features),
                num_features=len(todays_features.columns)
            )

            # Step 4: Load production model
            model_identifier = config.inference.model.identifier

            app_logger.structured_log(
                logging.INFO,
                "Loading production model",
                model_identifier=model_identifier
            )

            model_predictor = ModelPredictor(
                config=config,
                app_logger=app_logger,
                error_handler=error_handler,
                model_registry=model_registry
            )
            model_predictor.load_model(model_identifier)

            # Step 5: Make predictions (preprocessing happens automatically)
            app_logger.structured_log(
                logging.INFO,
                "Running inference",
                input_shape=todays_features.shape
            )

            raw_probabilities = model_predictor.predict(
                todays_features,
                return_probabilities=True
            )

            app_logger.structured_log(
                logging.INFO,
                "Raw predictions generated",
                num_predictions=len(raw_probabilities),
                prob_range=(float(np.min(raw_probabilities)), float(np.max(raw_probabilities)))
            )

            # Step 6: Apply postprocessing (calibration + conformal)
            calibrated_probs = raw_probabilities
            prediction_sets = None
            prediction_intervals = None

            # Load model artifacts (including postprocessing artifacts)
            model_data = model_registry.load_model(model_identifier)

            # Apply calibration if artifact exists and is enabled
            if config.inference.postprocessing.enable_calibration:
                calibrator_artifact = model_data.get('calibrator')

                if calibrator_artifact:
                    app_logger.structured_log(
                        logging.INFO,
                        "Applying probability calibration"
                    )

                    # Create calibrator and load artifact
                    calibrator = ProbabilityCalibrator(
                        app_logger=app_logger,
                        error_handler=error_handler,
                        method=calibrator_artifact.get('method', 'sigmoid')
                    )

                    # Load the fitted calibrator
                    calibrator.calibrator_ = calibrator_artifact.get('calibrator')
                    calibrator.calibration_metrics_ = calibrator_artifact.get('metrics', {})

                    # Transform probabilities
                    calibrated_probs = calibrator.transform(raw_probabilities)

                    app_logger.structured_log(
                        logging.INFO,
                        "Calibration applied",
                        method=calibrator_artifact.get('method'),
                        prob_shift=float(np.mean(calibrated_probs - raw_probabilities))
                    )
                elif not config.inference.postprocessing.skip_if_missing:
                    raise error_handler.create_error_handler(
                        'inference',
                        "Calibration enabled but artifact not found in model"
                    )
                else:
                    app_logger.structured_log(
                        logging.WARNING,
                        "Calibration enabled but artifact not found - using raw probabilities"
                    )

            # Apply conformal prediction if artifact exists and is enabled
            if config.inference.postprocessing.enable_conformal:
                conformal_artifact = model_data.get('conformal_predictor')

                if conformal_artifact:
                    app_logger.structured_log(
                        logging.INFO,
                        "Applying conformal prediction"
                    )

                    # Create conformal predictor and load artifact
                    conformal_predictor = ConformalPredictor(
                        app_logger=app_logger,
                        error_handler=error_handler,
                        method=conformal_artifact.get('method', 'split_conformal'),
                        alpha=conformal_artifact.get('alpha', 0.1)
                    )

                    # Load the fitted conformal predictor state
                    conformal_predictor.calibration_scores_ = conformal_artifact.get('calibration_scores')
                    conformal_predictor.alpha = conformal_artifact.get('alpha', 0.1)
                    conformal_predictor.method = conformal_artifact.get('method', 'split_conformal')

                    # Generate prediction sets and intervals
                    conformal_results = conformal_predictor.predict(
                        calibrated_probs,
                        return_sets=True,
                        return_intervals=True
                    )

                    prediction_sets = conformal_results.get('prediction_sets')
                    prediction_intervals = conformal_results.get('intervals')

                    app_logger.structured_log(
                        logging.INFO,
                        "Conformal prediction applied",
                        method=conformal_artifact.get('method'),
                        alpha=conformal_artifact.get('alpha'),
                        avg_interval_width=float(np.mean([
                            interval['upper'] - interval['lower']
                            for interval in prediction_intervals
                        ])) if prediction_intervals else None
                    )
                elif not config.inference.postprocessing.skip_if_missing:
                    raise error_handler.create_error_handler(
                        'inference',
                        "Conformal prediction enabled but artifact not found in model"
                    )
                else:
                    app_logger.structured_log(
                        logging.WARNING,
                        "Conformal prediction enabled but artifact not found - skipping"
                    )

            # Step 7: Format predictions
            predictions_df = _format_predictions(
                todays_features=todays_features,
                raw_probs=raw_probabilities,
                calibrated_probs=calibrated_probs,
                prediction_sets=prediction_sets,
                prediction_intervals=prediction_intervals,
                app_logger=app_logger,
                config=config
            )

            # Step 8: Save predictions
            predictions_dir = Path(config.inference.output.predictions_dir)
            predictions_dir.mkdir(parents=True, exist_ok=True)

            today_date = datetime.now().strftime('%Y-%m-%d')
            filename_pattern = config.inference.output.filename_pattern
            output_filename = filename_pattern.replace('{date}', today_date)
            output_path = predictions_dir / output_filename

            app_logger.structured_log(
                logging.INFO,
                "Saving predictions",
                output_path=str(output_path),
                num_predictions=len(predictions_df)
            )

            data_access.save_dataframes([predictions_df], [str(output_path)])

            # Log summary statistics
            avg_confidence = predictions_df['confidence'].mean() if 'confidence' in predictions_df.columns else None
            home_win_pct = (predictions_df['predicted_winner'] == 'home').mean() * 100 if 'predicted_winner' in predictions_df.columns else None

            app_logger.structured_log(
                logging.INFO,
                "Inference pipeline completed successfully",
                num_predictions=len(predictions_df),
                avg_confidence=float(avg_confidence) if avg_confidence else None,
                home_win_percentage=float(home_win_pct) if home_win_pct else None,
                output_file=str(output_path)
            )

    except Exception as e:
        # Check if it's one of our custom error types
        if hasattr(e, 'app_logger') and hasattr(e, 'exit_code'):
            _handle_known_error(app_logger, e)
        else:
            _handle_unexpected_error(app_logger, e)


def _format_predictions(todays_features: pd.DataFrame,
                       raw_probs: np.ndarray,
                       calibrated_probs: np.ndarray,
                       prediction_sets: Optional[list],
                       prediction_intervals: Optional[list],
                       app_logger,
                       config) -> pd.DataFrame:
    """
    Format predictions into output DataFrame.

    Args:
        todays_features: Engineered features for today's games
        raw_probs: Raw model probabilities
        calibrated_probs: Calibrated probabilities
        prediction_sets: Conformal prediction sets (optional)
        prediction_intervals: Conformal prediction intervals (optional)
        app_logger: Logger instance
        config: Configuration manager

    Returns:
        Formatted predictions DataFrame
    """
    # Extract game metadata from features
    # Note: The engineered data is in game-centric format (one row per game)
    # with home_ and visitor_ prefixed columns

    output_df = pd.DataFrame()

    # Game identifiers
    output_df['game_id'] = todays_features['game_id'].values
    output_df['game_date'] = todays_features['game_date'].values

    # Team names (extract from engineered features)
    # Assuming columns like 'team_id_home' and 'team_id_visitor' exist
    # Adjust based on your actual column names after merge_team_data
    if 'team_id_home' in todays_features.columns:
        output_df['home_team'] = todays_features['team_id_home'].values
        output_df['away_team'] = todays_features['team_id_visitor'].values
    elif 'team_home' in todays_features.columns:
        output_df['home_team'] = todays_features['team_home'].values
        output_df['away_team'] = todays_features['team_visitor'].values
    else:
        # Fallback: try to infer from available columns
        app_logger.structured_log(
            logging.WARNING,
            "Could not find standard team columns - predictions may be missing team info",
            available_columns=todays_features.columns.tolist()[:10]  # Log first 10 columns
        )
        output_df['home_team'] = 'UNKNOWN'
        output_df['away_team'] = 'UNKNOWN'

    # Probabilities
    if config.inference.output.include_raw_probabilities:
        output_df['raw_home_win_prob'] = raw_probs
    output_df['calibrated_home_win_prob'] = calibrated_probs

    # Predicted winner (based on calibrated probability)
    output_df['predicted_winner'] = output_df['calibrated_home_win_prob'].apply(
        lambda p: 'home' if p >= 0.5 else 'away'
    )

    # Confidence (max of home or away probability)
    output_df['confidence'] = output_df['calibrated_home_win_prob'].apply(
        lambda p: max(p, 1 - p)
    )

    # Add conformal prediction outputs if available
    if prediction_sets is not None and config.inference.output.include_prediction_sets:
        output_df['prediction_set'] = [str(ps) for ps in prediction_sets]

    if prediction_intervals is not None and config.inference.output.include_probability_intervals:
        output_df['prob_lower'] = [interval['lower'] for interval in prediction_intervals]
        output_df['prob_upper'] = [interval['upper'] for interval in prediction_intervals]
        output_df['interval_width'] = output_df['prob_upper'] - output_df['prob_lower']

    # Metadata
    output_df['prediction_timestamp'] = datetime.now().isoformat()

    if config.inference.output.include_model_metadata:
        output_df['model_identifier'] = config.inference.model.identifier

    return output_df


def _handle_known_error(app_logger, e):
    """Handle known custom errors"""
    if app_logger:
        app_logger.structured_log(
            logging.ERROR,
            f"{type(e).__name__} occurred",
            error_message=str(e),
            error_type=type(e).__name__,
            traceback=traceback.format_exc()
        )
    sys.exit(type(e).exit_code)


def _handle_unexpected_error(app_logger, e):
    """Handle unexpected errors"""
    if app_logger:
        app_logger.structured_log(
            logging.CRITICAL,
            "Unexpected error occurred",
            error_message=str(e),
            error_type=type(e).__name__,
            traceback=traceback.format_exc()
        )
    else:
        print(f"CRITICAL: Unexpected error occurred: {str(e)}")
        print(traceback.format_exc())
    sys.exit(6)


if __name__ == "__main__":
    main()
