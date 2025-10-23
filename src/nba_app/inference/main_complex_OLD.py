"""Main entry point for NBA game prediction pipeline.

This module runs daily to generate predictions for upcoming NBA games.

Pipeline:
1. Load tomorrow's scheduled games
2. Load historical data for feature engineering
3. Generate predictions with uncertainty quantification
4. Save predictions for dashboard consumption

Usage:
    python -m src.nba_app.inference.main
    or
    uv run -m src.nba_app.inference.main
"""

import sys
import traceback
import logging
from datetime import datetime, timedelta
from pathlib import Path

from .di_container import DIContainer

LOG_FILE = "inference.log"


def main() -> None:
    """
    Main function to run the NBA game prediction pipeline.
    """

    container = DIContainer()
    app_logger = None

    try:
        config = container.config()

        # Setup the app logger
        app_logger = container.app_logger()
        app_logger.setup(LOG_FILE)

        schedule_loader = container.schedule_loader()
        game_predictor = container.game_predictor()
        data_access = container.data_access()
        error_handler = container.error_handler_factory()

        app_logger.structured_log(
            logging.INFO,
            "Starting NBA inference pipeline",
            app_version=config.app_version,
            environment=config.environment,
            log_level=config.core.app_logging_config.log_level if hasattr(config, 'core') else 'INFO'
        )

        with app_logger.log_context(app_version=config.app_version, environment=config.environment):

            # Load production model
            app_logger.structured_log(logging.INFO, "Loading production model")
            game_predictor.load_production_model()

            # Load tomorrow's schedule
            tomorrow = datetime.now() + timedelta(days=1)
            app_logger.structured_log(
                logging.INFO,
                "Loading scheduled games",
                target_date=tomorrow.strftime('%Y-%m-%d')
            )

            schedule_df = schedule_loader.load_upcoming_games(target_date=tomorrow)

            if schedule_df.empty:
                app_logger.structured_log(
                    logging.INFO,
                    "No games scheduled for tomorrow - nothing to predict"
                )
                return

            # Validate schedule
            app_logger.structured_log(logging.INFO, "Validating schedule")
            schedule_loader.validate_schedule(schedule_df)

            # Load historical data for feature engineering
            app_logger.structured_log(logging.INFO, "Loading historical data")
            historical_path = config.inference.data_sources.historical_boxscores

            historical_df = data_access.load_dataframe(historical_path)

            # Filter to recent data (lookback window)
            lookback_days = config.inference.data_sources.lookback_days
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            cutoff_str = cutoff_date.strftime('%Y-%m-%d')

            if 'game_date' in historical_df.columns:
                historical_df['game_date_parsed'] = pd.to_datetime(historical_df['game_date'])
                historical_df = historical_df[historical_df['game_date_parsed'] >= cutoff_str]

            app_logger.structured_log(
                logging.INFO,
                "Historical data loaded",
                num_games=len(historical_df),
                lookback_days=lookback_days
            )

            # Generate predictions
            app_logger.structured_log(
                logging.INFO,
                "Generating predictions",
                num_games=len(schedule_df)
            )

            predictions_df = game_predictor.predict_games(schedule_df, historical_df)

            # Save predictions
            predictions_dir = Path(config.inference.output.predictions_dir)
            predictions_dir.mkdir(parents=True, exist_ok=True)

            filename_pattern = config.inference.output.filename_pattern
            prediction_date = tomorrow.strftime('%Y-%m-%d')
            output_filename = filename_pattern.replace('{date}', prediction_date)
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
