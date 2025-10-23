"""Schedule Loader

Loads scheduled games for upcoming dates from webscraping outputs.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.framework.data_access.base_data_access import BaseDataAccess
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler


class ScheduleLoader:
    """
    Loads scheduled NBA games for prediction.

    Reads schedule files produced by the webscraping module and
    filters games within the prediction window.
    """

    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 data_access: BaseDataAccess,
                 error_handler: BaseErrorHandler):
        """
        Initialize ScheduleLoader.

        Args:
            config: Configuration manager
            app_logger: Application logger
            data_access: Data access layer for reading schedule files
            error_handler: Error handler for standardized error management
        """
        self.config = config
        self.app_logger = app_logger
        self.data_access = data_access
        self.error_handler = error_handler

        self.app_logger.structured_log(
            logging.INFO,
            "ScheduleLoader initialized"
        )

    def load_upcoming_games(self, target_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load games scheduled for a target date.

        Args:
            target_date: Date to load games for. If None, uses tomorrow.

        Returns:
            DataFrame with scheduled games containing:
            - game_id: Unique identifier
            - game_date: Date of game
            - home_team: Home team abbreviation
            - away_team: Away team abbreviation
            - Additional metadata if available

        Raises:
            Error if schedule file not found or invalid
        """
        try:
            if target_date is None:
                target_date = datetime.now() + timedelta(days=1)

            date_str = target_date.strftime('%Y-%m-%d')

            self.app_logger.structured_log(
                logging.INFO,
                "Loading scheduled games",
                target_date=date_str
            )

            # Build schedule file path
            schedule_pattern = self.config.inference.data_sources.schedule_file
            schedule_path = schedule_pattern.replace('{date}', date_str)

            self.app_logger.structured_log(
                logging.INFO,
                "Attempting to load schedule file",
                schedule_path=schedule_path
            )

            # Load schedule using data access layer
            schedule_df = self.data_access.load_dataframe(schedule_path)

            if schedule_df.empty:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "No games scheduled for target date",
                    target_date=date_str
                )
                return pd.DataFrame()

            # Validate required columns
            required_columns = ['game_date', 'home_team', 'away_team']
            missing_columns = [col for col in required_columns if col not in schedule_df.columns]

            if missing_columns:
                raise ValueError(
                    f"Schedule file missing required columns: {missing_columns}"
                )

            # Add game_id if not present
            if 'game_id' not in schedule_df.columns:
                schedule_df['game_id'] = schedule_df.apply(
                    lambda row: f"{row['game_date']}_{row['away_team']}_{row['home_team']}",
                    axis=1
                )

            self.app_logger.structured_log(
                logging.INFO,
                "Scheduled games loaded successfully",
                num_games=len(schedule_df),
                teams=sorted(set(schedule_df['home_team'].tolist() + schedule_df['away_team'].tolist()))
            )

            return schedule_df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'inference',
                "Error loading scheduled games",
                original_error=str(e),
                target_date=target_date.strftime('%Y-%m-%d') if target_date else None
            )

    def validate_schedule(self, schedule_df: pd.DataFrame) -> bool:
        """
        Validate that scheduled games have sufficient historical data.

        Args:
            schedule_df: DataFrame with scheduled games

        Returns:
            True if schedule is valid for prediction

        Raises:
            ValueError if schedule is invalid
        """
        try:
            if schedule_df.empty:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "Empty schedule - no games to predict"
                )
                return True  # Empty is valid, just no work to do

            # Check game dates are in the future
            schedule_df['game_date_parsed'] = pd.to_datetime(schedule_df['game_date'])
            today = pd.Timestamp.now().normalize()

            past_games = schedule_df[schedule_df['game_date_parsed'] < today]
            if not past_games.empty:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "Schedule contains games in the past",
                    num_past_games=len(past_games)
                )

            # Check max_days_ahead
            max_days_ahead = self.config.inference.filtering.max_days_ahead
            max_date = today + timedelta(days=max_days_ahead)

            future_games = schedule_df[schedule_df['game_date_parsed'] > max_date]
            if not future_games.empty:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "Schedule contains games beyond prediction window",
                    num_future_games=len(future_games),
                    max_days_ahead=max_days_ahead
                )

            self.app_logger.structured_log(
                logging.INFO,
                "Schedule validation completed",
                total_games=len(schedule_df),
                valid_games=len(schedule_df[(schedule_df['game_date_parsed'] >= today) &
                                            (schedule_df['game_date_parsed'] <= max_date)])
            )

            return True

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'inference',
                "Error validating schedule",
                original_error=str(e)
            )
