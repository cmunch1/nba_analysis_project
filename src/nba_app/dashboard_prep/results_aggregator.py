"""Results Aggregator

Loads yesterday's game results and validates against predictions.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.framework.data_access.base_data_access import BaseDataAccess
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler


class ResultsAggregator:
    """
    Aggregates actual game results and validates against predictions.

    Loads completed games from cumulative scraped data and matches them
    with predictions to calculate accuracy.
    """

    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 data_access: BaseDataAccess,
                 error_handler: BaseErrorHandler):
        """
        Initialize ResultsAggregator.

        Args:
            config: Configuration manager
            app_logger: Application logger
            data_access: Data access layer
            error_handler: Error handler
        """
        self.config = config
        self.app_logger = app_logger
        self.data_access = data_access
        self.error_handler = error_handler

        self.app_logger.structured_log(
            logging.INFO,
            "ResultsAggregator initialized"
        )

    def load_recent_results(self, lookback_days: Optional[int] = None) -> pd.DataFrame:
        """
        Load recent game results from cumulative scraped data.

        Args:
            lookback_days: Number of days back to load. If None, uses config.

        Returns:
            DataFrame with recent completed games

        Raises:
            Error if results cannot be loaded
        """
        try:
            if lookback_days is None:
                lookback_days = self.config.dashboard_prep.results.lookback_days

            self.app_logger.structured_log(
                logging.INFO,
                "Loading recent game results",
                lookback_days=lookback_days
            )

            # Load cumulative box scores using pandas directly (data_access adds directory prefix)
            results_path = self.config.dashboard_prep.input_paths.actual_results
            results_df = pd.read_csv(results_path)

            if results_df.empty:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "No game results found"
                )
                return pd.DataFrame()

            # Filter to recent games
            cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

            if 'game_date' in results_df.columns:
                results_df['game_date_parsed'] = pd.to_datetime(results_df['game_date'])
                recent_results = results_df[results_df['game_date_parsed'] >= cutoff_date]
            else:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "game_date column not found - returning all results"
                )
                recent_results = results_df

            self.app_logger.structured_log(
                logging.INFO,
                "Recent results loaded",
                num_games=len(recent_results),
                lookback_days=lookback_days
            )

            return recent_results

        except Exception as e:
            self.app_logger.structured_log(
                logging.WARNING,
                "Could not load recent results",
                error=str(e)
            )
            return pd.DataFrame()

    def validate_against_predictions(self,
                                    results_df: pd.DataFrame,
                                    predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate actual results against predictions.

        Args:
            results_df: DataFrame with actual game results (team-centric)
            predictions_df: DataFrame with predictions (game-centric)

        Returns:
            DataFrame with results + prediction validation
        """
        try:
            if results_df.empty or predictions_df.empty:
                self.app_logger.structured_log(
                    logging.INFO,
                    "No results or predictions to validate"
                )
                return pd.DataFrame()

            self.app_logger.structured_log(
                logging.INFO,
                "Validating results against predictions",
                num_results=len(results_df),
                num_predictions=len(predictions_df)
            )

            # Convert results from team-centric to game-centric format
            game_results = self._create_game_results(results_df)

            # Merge with predictions
            validated_df = pd.merge(
                game_results,
                predictions_df,
                on='game_id',
                how='inner',
                suffixes=('_actual', '_predicted')
            )

            if validated_df.empty:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "No matching games found between results and predictions"
                )
                return pd.DataFrame()

            # Calculate prediction accuracy
            validated_df['prediction_correct'] = (
                validated_df['actual_winner'] == validated_df['predicted_winner']
            )

            # Calculate prediction error (for probabilities)
            if 'calibrated_home_win_prob' in validated_df.columns:
                # Convert actual winner to binary (1 = home win, 0 = away win)
                validated_df['actual_home_win'] = (
                    validated_df['actual_winner'] == 'home'
                ).astype(int)

                # Prediction error is |predicted_prob - actual_outcome|
                validated_df['prediction_error'] = np.abs(
                    validated_df['calibrated_home_win_prob'] - validated_df['actual_home_win']
                )

            self.app_logger.structured_log(
                logging.INFO,
                "Results validated against predictions",
                num_validated=len(validated_df),
                accuracy=float(validated_df['prediction_correct'].mean()) if 'prediction_correct' in validated_df.columns else None
            )

            return validated_df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_processing',
                "Error validating results against predictions",
                original_error=str(e)
            )

    def _create_game_results(self, team_centric_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert team-centric results to game-centric format.

        Args:
            team_centric_df: Team-centric results (2 rows per game)

        Returns:
            Game-centric DataFrame (1 row per game)
        """
        try:
            # Split into home and away teams
            home_df = team_centric_df[team_centric_df['is_home_team'] == 1].copy()
            away_df = team_centric_df[team_centric_df['is_home_team'] == 0].copy()

            # Merge to create game-centric view
            games_df = pd.merge(
                home_df[['game_id', 'game_date', 'team_id', 'team', 'pts', 'is_win']],
                away_df[['game_id', 'team_id', 'team', 'pts']],
                on='game_id',
                suffixes=('_home', '_away')
            )

            # Rename columns
            games_df = games_df.rename(columns={
                'team_home': 'home_team',
                'team_away': 'away_team',
                'pts_home': 'actual_home_score',
                'pts_away': 'actual_away_score'
            })

            # Determine actual winner
            games_df['actual_winner'] = games_df.apply(
                lambda row: 'home' if row['is_win'] == 1 else 'away',
                axis=1
            )

            # Keep only needed columns
            games_df = games_df[[
                'game_id',
                'game_date',
                'home_team',
                'away_team',
                'actual_winner',
                'actual_home_score',
                'actual_away_score'
            ]]

            return games_df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_processing',
                "Error creating game results from team-centric data",
                original_error=str(e)
            )

    def format_for_dashboard(self, validated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Format validated results for dashboard display.

        Args:
            validated_df: Validated results DataFrame

        Returns:
            Formatted DataFrame for dashboard
        """
        try:
            if validated_df.empty:
                return pd.DataFrame()

            self.app_logger.structured_log(
                logging.INFO,
                "Formatting results for dashboard",
                num_results=len(validated_df)
            )

            # Select columns for dashboard
            include_columns = self.config.dashboard_prep.results.include_columns

            # Keep only columns that exist
            available_columns = [col for col in include_columns if col in validated_df.columns]

            if available_columns:
                dashboard_df = validated_df[available_columns].copy()
            else:
                dashboard_df = validated_df.copy()

            # Add section identifier
            dashboard_df['section'] = 'results'

            # Sort by game date (most recent first)
            if 'game_date' in dashboard_df.columns:
                dashboard_df = dashboard_df.sort_values('game_date', ascending=False)

            self.app_logger.structured_log(
                logging.INFO,
                "Results formatted for dashboard",
                num_rows=len(dashboard_df)
            )

            return dashboard_df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_processing',
                "Error formatting results for dashboard",
                original_error=str(e)
            )
