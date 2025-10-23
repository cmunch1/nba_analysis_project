"""Main entry point for Dashboard Data Preparation

This module runs daily AFTER inference to aggregate all data for the dashboard.

Pipeline:
1. Load tomorrow's predictions
2. Load and validate yesterday's results
3. Calculate performance metrics (season, 7-day, 30-day)
4. Calculate team-level performance
5. Generate drift monitoring data
6. Combine into single denormalized CSV
7. Save to data/dashboard/dashboard_data.csv

Usage:
    python -m src.nba_app.dashboard_prep.main
    or
    uv run -m src.nba_app.dashboard_prep.main
"""

import sys
import traceback
import logging

from .di_container import DIContainer

LOG_FILE = "dashboard_prep.log"


def main() -> None:
    """
    Main function to generate dashboard data.
    """

    container = DIContainer()
    app_logger = None

    try:
        config = container.config()

        # Setup the app logger
        app_logger = container.app_logger()
        app_logger.setup(LOG_FILE)

        dashboard_data_generator = container.dashboard_data_generator()
        error_handler = container.error_handler_factory()

        app_logger.structured_log(
            logging.INFO,
            "Starting dashboard data preparation",
            app_version=config.app_version,
            environment=config.environment,
            log_level=config.core.app_logging_config.log_level if hasattr(config, 'core') else 'INFO'
        )

        with app_logger.log_context(app_version=config.app_version, environment=config.environment):

            # Generate dashboard data
            app_logger.structured_log(logging.INFO, "Generating dashboard data from all sources")
            dashboard_df = dashboard_data_generator.generate_dashboard_data()

            if dashboard_df.empty:
                app_logger.structured_log(
                    logging.WARNING,
                    "No dashboard data generated - output will be empty"
                )
                # Don't fail - dashboard can handle empty data
                return

            # Save dashboard data
            app_logger.structured_log(
                logging.INFO,
                "Saving dashboard data",
                num_rows=len(dashboard_df)
            )

            output_path = dashboard_data_generator.save_dashboard_data(dashboard_df)

            # Log summary statistics
            sections = dashboard_df['section'].unique() if 'section' in dashboard_df.columns else []
            section_counts = {}
            for section in sections:
                section_counts[section] = int((dashboard_df['section'] == section).sum())

            app_logger.structured_log(
                logging.INFO,
                "Dashboard data preparation completed successfully",
                output_path=output_path,
                total_rows=len(dashboard_df),
                sections=section_counts
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
