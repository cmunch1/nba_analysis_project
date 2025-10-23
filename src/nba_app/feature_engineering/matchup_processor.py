"""Matchup Processor

Creates placeholder rows for today's scheduled games.
These rows have zero values for all stats (since games haven't been played yet),
but allow feature engineering to calculate rolling averages from historical data.
"""

import pandas as pd
import logging
from datetime import datetime
from typing import Optional, Dict

from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.framework.data_access.base_data_access import BaseDataAccess
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler


class MatchupProcessor:
    """
    Processes today's matchups to create placeholder rows for feature engineering.

    Loads todays_matchups.csv and todays_games_ids.csv from webscraping output,
    and creates rows matching the schema of processed team-centric data.
    Uses team_mapping.yaml to convert team IDs to abbreviations.
    """

    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 data_access: BaseDataAccess,
                 error_handler: BaseErrorHandler):
        """
        Initialize MatchupProcessor.

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

        # Load team ID to abbreviation mapping
        self.team_id_to_abbrev = self._load_team_mapping()

        self.app_logger.structured_log(
            logging.INFO,
            "MatchupProcessor initialized",
            num_teams_mapped=len(self.team_id_to_abbrev)
        )

    def _load_team_mapping(self) -> Dict[str, str]:
        """
        Load team ID to abbreviation mapping from config.

        Returns:
            Dictionary mapping team IDs (as strings) to abbreviations
        """
        try:
            # Access team mapping from config
            if hasattr(self.config, 'team_id_to_abbrev'):
                team_mapping = self.config.team_id_to_abbrev

                # Convert SimpleNamespace to dict if needed
                if hasattr(team_mapping, '__dict__'):
                    team_mapping = vars(team_mapping)

                return team_mapping
            else:
                return {}
        except Exception as e:
            # Logger might not be initialized yet, return empty dict
            return {}

    def load_todays_matchups(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load today's matchups and game IDs from webscraping output.

        Returns:
            Tuple of (matchups_df, games_df)
            - matchups_df: DataFrame with columns [visitor_id, home_id]
            - games_df: DataFrame with column [game_id]

        Raises:
            Error if files not found or invalid
        """
        try:
            self.app_logger.structured_log(
                logging.INFO,
                "Loading today's matchups from webscraping output"
            )

            # Load matchups (visitor_id, home_id)
            # These files are in newly_scraped directory, not cumulative
            matchups_path = f"data/newly_scraped/{self.config.todays_matchups_file}"
            matchups_df = pd.read_csv(matchups_path)

            # Load game IDs
            games_path = f"data/newly_scraped/{self.config.todays_games_ids_file}"
            games_df = pd.read_csv(games_path)

            if matchups_df.empty or games_df.empty:
                self.app_logger.structured_log(
                    logging.INFO,
                    "No games scheduled for today"
                )
                return pd.DataFrame(), pd.DataFrame()

            if len(matchups_df) != len(games_df):
                raise ValueError(
                    f"Mismatch between matchups ({len(matchups_df)}) "
                    f"and game IDs ({len(games_df)})"
                )

            self.app_logger.structured_log(
                logging.INFO,
                "Today's matchups loaded successfully",
                num_games=len(matchups_df)
            )

            return matchups_df, games_df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_storage',
                "Error loading today's matchups",
                original_error=str(e)
            )

    def create_placeholder_rows(self,
                               matchups_df: pd.DataFrame,
                               games_df: pd.DataFrame,
                               reference_df: pd.DataFrame,
                               today_date: Optional[str] = None) -> pd.DataFrame:
        """
        Create placeholder rows for today's games matching processed data schema.

        Args:
            matchups_df: DataFrame with columns [visitor_id, home_id]
            games_df: DataFrame with column [game_id]
            reference_df: Reference DataFrame to get column structure and dtypes
            today_date: Date string (YYYY-MM-DD). If None, uses today.

        Returns:
            DataFrame with placeholder rows (two rows per game: home and away)
            All stat columns set to 0, only identifiers populated

        Raises:
            Error if placeholder creation fails
        """
        try:
            if matchups_df.empty or games_df.empty:
                self.app_logger.structured_log(
                    logging.INFO,
                    "No matchups to process - returning empty DataFrame"
                )
                return pd.DataFrame(columns=reference_df.columns)

            if today_date is None:
                today_date = datetime.now().strftime('%Y-%m-%d')

            self.app_logger.structured_log(
                logging.INFO,
                "Creating placeholder rows for today's games",
                num_games=len(games_df),
                date=today_date
            )

            # Combine matchups with game IDs
            combined = pd.concat([matchups_df.reset_index(drop=True),
                                 games_df.reset_index(drop=True)], axis=1)

            placeholder_rows = []

            for _, row in combined.iterrows():
                game_id = row['game_id']
                home_id = row['home_id']
                visitor_id = row['visitor_id']

                # Create home team row
                home_row = self._create_single_placeholder_row(
                    game_id=game_id,
                    team_id=home_id,
                    opponent_id=visitor_id,
                    is_home=True,
                    game_date=today_date,
                    reference_df=reference_df
                )
                placeholder_rows.append(home_row)

                # Create away team row
                away_row = self._create_single_placeholder_row(
                    game_id=game_id,
                    team_id=visitor_id,
                    opponent_id=home_id,
                    is_home=False,
                    game_date=today_date,
                    reference_df=reference_df
                )
                placeholder_rows.append(away_row)

            # Combine all placeholder rows
            placeholder_df = pd.DataFrame(placeholder_rows, columns=reference_df.columns)

            # Ensure correct dtypes match reference
            for col in placeholder_df.columns:
                if col in reference_df.columns:
                    try:
                        placeholder_df[col] = placeholder_df[col].astype(reference_df[col].dtype)
                    except Exception:
                        # If type conversion fails, keep as-is
                        pass

            self.app_logger.structured_log(
                logging.INFO,
                "Placeholder rows created successfully",
                num_rows=len(placeholder_df),
                num_games=len(games_df)
            )

            return placeholder_df

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'data_processing',
                "Error creating placeholder rows",
                original_error=str(e)
            )

    def _create_single_placeholder_row(self,
                                       game_id: str,
                                       team_id: str,
                                       opponent_id: str,
                                       is_home: bool,
                                       game_date: str,
                                       reference_df: pd.DataFrame) -> dict:
        """
        Create a single placeholder row for one team in one game.

        Args:
            game_id: NBA game ID
            team_id: Team ID
            opponent_id: Opponent team ID
            is_home: True if home team
            game_date: Game date string
            reference_df: Reference DataFrame for column structure

        Returns:
            Dictionary representing one row
        """
        # Get current season (derive from date)
        year = int(game_date.split('-')[0])
        month = int(game_date.split('-')[1])
        # NBA season spans two calendar years (starts in Oct)
        season = year if month >= 10 else year - 1

        # Initialize row with zeros for all columns
        row = {col: 0 for col in reference_df.columns}

        # Populate identifiers and metadata
        row['game_id'] = game_id
        row['season'] = season
        row['game_date'] = game_date
        row['team_id'] = team_id
        row['is_home_team'] = 1 if is_home else 0

        # Set playoff/overtime flags (assume regular season, no overtime)
        row['is_playoff'] = 0
        row['is_overtime'] = 0

        # Set sub_season_id (2 = regular season)
        row['sub_season_id'] = 2

        # Map team IDs to abbreviations
        team_abbrev = self.team_id_to_abbrev.get(str(team_id), str(team_id))
        opponent_abbrev = self.team_id_to_abbrev.get(str(opponent_id), str(opponent_id))

        # Create match_up string (format: "TEAM @ OPP" or "TEAM vs. OPP")
        if is_home:
            row['match_up'] = f"{team_abbrev} vs. {opponent_abbrev}"
            row['team'] = team_abbrev
        else:
            row['match_up'] = f"{team_abbrev} @ {opponent_abbrev}"
            row['team'] = team_abbrev

        # Minutes played - set to expected game length
        row['min'] = 48  # Standard NBA game is 48 minutes

        # All stats remain 0 (game hasn't been played)
        # This includes: pts, fgm, fga, rebounds, etc.

        # Special handling for percentage fields - set to None/NaN instead of 0
        # to avoid division-by-zero issues in feature engineering
        percentage_cols = ['fg_pct', '3p_pct', 'ft_pct', 'oreb_pct', 'dreb_pct',
                          'reb_pct', 'ast_pct', 'tov_pct', 'efg_pct', 'ts_pct']
        for col in percentage_cols:
            if col in row:
                row[col] = None

        return row
