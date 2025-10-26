"""
    Feature engineering 
        - rolling averages of key stats, (e.g.free throw % for last 3 games, 5 games, and 10 games)
        - win/lose streaks (e.g. -3 means lost 3 games in a row)
        - home/visitor streaks (e.g. -3 means played 3 games in a row on the road)
        - specific matchup (team X vs team Y) rolling averages and streaks
        - league average rolling stats
        - elo ratings


"""

import pandas as pd
import numpy as np
import logging
import yaml
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from sklearn.model_selection import train_test_split
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.error_handler import FeatureEngineeringError
from .base_feature_engineering import BaseFeatureEngineer
from .feature_schema import FeatureSchema


class FeatureEngineer(BaseFeatureEngineer):
    def __init__(self, config: BaseConfigManager, app_logger: BaseAppLogger):
        """
        Initialize the FeatureEngineer class.

        Args:
            config (BaseConfigManager): Configuration object containing feature engineering parameters.
            app_logger (BaseAppLogger): Logger instance for error handling.
        """
        self.config = config
        self.app_logger = app_logger
        self.feature_allowlist = None

        # Load feature allowlist if enabled
        if hasattr(config, 'feature_allowlist') and getattr(config.feature_allowlist, 'enabled', False):
            self._load_feature_allowlist()

        self.app_logger.structured_log(
            logging.INFO,
            "FeatureEngineer initialized",
            config_type=type(config).__name__,
            allowlist_enabled=self.feature_allowlist is not None,
            allowlist_size=len(self.feature_allowlist) if self.feature_allowlist else 0
        )

    @staticmethod
    def log_performance(func):
        """Decorator factory for performance logging"""
        def wrapper(*args, **kwargs):
            # Get the self instance from args since this is now a static method
            instance = args[0]
            return instance.app_logger.log_performance(func)(*args, **kwargs)
        return wrapper

    def _load_feature_allowlist(self) -> None:
        """
        Load feature allowlist from YAML file if enabled in config.
        The allowlist will be used to filter which features are calculated.
        """
        try:
            allowlist_file = Path(self.config.feature_allowlist.allowlist_file)

            if not allowlist_file.exists():
                self.app_logger.structured_log(
                    logging.WARNING,
                    "Feature allowlist file not found, proceeding with all features",
                    allowlist_file=str(allowlist_file)
                )
                return

            with open(allowlist_file, 'r') as f:
                allowlist_data = yaml.safe_load(f)

            if 'feature_allowlist' not in allowlist_data:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "Invalid allowlist file format, missing 'feature_allowlist' key",
                    allowlist_file=str(allowlist_file)
                )
                return

            features = allowlist_data['feature_allowlist'].get('features', [])

            # Store as a set for fast lookups
            self.feature_allowlist = set(features)

            # Log metadata if available
            metadata = allowlist_data['feature_allowlist'].get('metadata', {})

            self.app_logger.structured_log(
                logging.INFO,
                "Feature allowlist loaded successfully",
                allowlist_file=str(allowlist_file),
                num_features=len(self.feature_allowlist),
                source_audit=metadata.get('source_audit'),
                recommendation=metadata.get('recommendation')
            )

        except Exception as e:
            self.app_logger.structured_log(
                logging.ERROR,
                "Failed to load feature allowlist, proceeding with all features",
                error=str(e)
            )
            self.feature_allowlist = None

    def _filter_stats_for_allowlist(self, stats_columns: Set[str], periods: List[int], suffix: str) -> Set[str]:
        """
        Filter stats columns to only include those that will produce features in the allowlist.

        The allowlist contains final feature names like 'h_2nd_pts_rolling_avg_10_all'.
        We need to work backwards to determine which base stats (like '2nd_pts') should be kept.

        Args:
            stats_columns (Set[str]): Original set of stats columns to filter
            periods (List[int]): List of rolling average periods
            suffix (str): Suffix for this run (all, home, visitor)

        Returns:
            Set[str]: Filtered set of stats columns
        """
        if self.feature_allowlist is None:
            return stats_columns

        # Build a set of base stats that are needed
        needed_stats = set()

        # For each stat column, check if any of its generated features are in the allowlist
        for stat_col in stats_columns:
            for period in periods:
                # Generate what the feature name would be after processing
                # Pattern: {stat_col}_rolling_avg_{period}_{suffix}
                # After merge_team_data, it gets prefixed with h_ or v_
                feature_name_base = f"{stat_col}_rolling_avg_{period}_{suffix}"

                # Check if either the home or visitor version is in allowlist
                if (f"{self.config.home_team_prefix}{feature_name_base}" in self.feature_allowlist or
                    f"{self.config.visitor_team_prefix}{feature_name_base}" in self.feature_allowlist):
                    needed_stats.add(stat_col)
                    break  # No need to check other periods for this stat

        return needed_stats

    @log_performance
    def engineer_features(self, df: pd.DataFrame, export_schema: bool = False) -> pd.DataFrame:
        """
        Engineer features for the input dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.
            export_schema (bool): Whether to export feature schema after engineering.

        Returns:
            pd.DataFrame: Dataframe with engineered features.

        Raises:
            FeatureEngineeringError: If there's an error during feature engineering.
        """
        self.app_logger.structured_log(
            logging.INFO,
            "Starting feature engineering",
            input_shape=df.shape,
            export_schema=export_schema
        )
        try:
            if self.config.include_postseason:
                df = df[df[self.config.is_playoff_column] == False]

            # cycle through home, visitor, and combined cases to create rolling averages, win streaks, etc.
            df_merged = df.copy()
            suffixes = [self.config.combined_cases_suffix, self.config.home_game_suffix, self.config.visitor_game_suffix]

            for suffix in suffixes:
                if suffix == self.config.combined_cases_suffix:
                    df_working = df.copy()
                else:
                    df_working = df[df[self.config.home_team_column] == (suffix == self.config.home_game_suffix)].copy()

                # rolling averages from previous sets of games (e.g. last 3 games, 5 games, 10 games)
                df_rolling_avgs = self._create_rolling_averages(df_working, suffix)
                merge_columns = [col for col in df_rolling_avgs.columns
                                 if "rolling_avg" in col]
                df_merged = self._merge_features(df_merged, df_rolling_avgs, merge_columns, suffix)

                # games in a row winning or losing
                df_streaks = self._calculate_win_lose_streaks(df_working, suffix)
                df_merged = self._merge_features(df_merged, df_streaks, [f'win_streak_{suffix}'], suffix)

            # games in a row as home or visitor
            df_merged = self._calculate_home_visitor_streaks(df_merged)

            # elo ratings
            df_merged = self._update_elo_ratings(df_merged)

            # free up memory
            del df_rolling_avgs
            del df_working
            del df_streaks
            del df

            # Export feature schema if requested
            if export_schema:
                self._export_feature_schema(df_merged)

            # Sort columns alphabetically to ensure deterministic ordering
            # Keep key metadata columns at the front, then sort the rest
            metadata_cols = ['game_id', 'season', 'game_date', 'sub_season_id',
                           'h_team', 'v_team', 'h_match_up', 'v_match_up', 'h_is_win']

            # Get columns that exist in the DataFrame
            existing_metadata = [col for col in metadata_cols if col in df_merged.columns]

            # Get remaining columns and sort them
            remaining_cols = [col for col in df_merged.columns if col not in existing_metadata]
            sorted_remaining = sorted(remaining_cols)

            # Reorder: metadata first, then sorted feature columns
            ordered_cols = existing_metadata + sorted_remaining
            df_merged = df_merged[ordered_cols]

            self.app_logger.structured_log(
                logging.INFO,
                "Feature engineering completed with deterministic column ordering",
                output_shape=df_merged.shape,
                num_metadata_cols=len(existing_metadata),
                num_feature_cols=len(sorted_remaining)
            )

            return df_merged

        except Exception as e:

            raise FeatureEngineeringError("Error in feature engineering",
                                          self.app_logger,
                                          error_message=str(e),
                                          dataframe_shape=df.shape)


        
    def _create_rolling_averages(self, df: pd.DataFrame, home_or_visitor_suffix: str) -> pd.DataFrame:
        """
        Create rolling averages for the input dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.
            home_or_visitor_suffix (str): Suffix for the feature names (all, home, or visitor).

        Returns:
            pd.DataFrame: Dataframe with rolling averages.
        """
        self.app_logger.structured_log(logging.INFO, "Creating rolling averages -" + home_or_visitor_suffix)
        all_columns = set(df.columns)
        stats_columns = all_columns.difference(self.config.game_info_columns + self.config.team_info_columns)
        periods = self.config.team_rolling_avg_periods

        # Filter stats_columns based on feature allowlist if enabled
        if self.feature_allowlist is not None:
            stats_columns = self._filter_stats_for_allowlist(stats_columns, periods, home_or_visitor_suffix)
            self.app_logger.structured_log(
                logging.INFO,
                "Filtered stats columns based on allowlist",
                suffix=home_or_visitor_suffix,
                original_count=len(all_columns.difference(self.config.game_info_columns + self.config.team_info_columns)),
                filtered_count=len(stats_columns)
            )
        
        df = df.sort_values(by = [self.config.new_date_column, self.config.new_game_id_column], axis=0, ascending=[True, True,], ignore_index=True)

        
        result_df = df.copy()
        
        stats_columns = list(stats_columns)
        
        for period in periods:
            if self.config.extend_rolling_avgs:
                # Calculate rolling average across all seasons, but specific to each team
                rolling_avgs = df.groupby(self.config.new_team_id_column)[stats_columns].rolling(
                    window=period+1, min_periods=1
                ).mean().shift(1).reset_index(level=0, drop=True)
            else:
                # Calculate rolling average within each season and for each team
                rolling_avgs = df.groupby([self.config.new_team_id_column, self.config.season_column])[stats_columns].rolling(
                    window=period+1, min_periods=1
                ).mean().shift(1).reset_index(level=[0,1], drop=True)
                
                # Apply the condition to set NaN for the first 'period' rows of each group
                mask = df.groupby([self.config.new_team_id_column, self.config.season_column]).cumcount() < period
                rolling_avgs[mask] = np.nan

            # Rename columns to include period and suffix
            rolling_avgs.columns = [f"{col}_rolling_avg_{period}_{home_or_visitor_suffix}" for col in rolling_avgs.columns]
            
            # Join the rolling averages to the result dataframe
            result_df = result_df.join(rolling_avgs)

        # drop stats_columns - they were used to calculate rolling averages, but are no longer needed
        result_df = result_df.drop(columns=stats_columns)

        self.app_logger.structured_log(
            logging.INFO,
            "Rolling averages created",
            output_shape=result_df.shape
        )
        
        return result_df

    def _calculate_win_lose_streaks(self, df: pd.DataFrame, home_or_visitor_suffix: str) -> pd.DataFrame:
        """
        Calculate a single streak for both home and visitor teams.
        Positive values indicate winning streaks, negative values indicate losing streaks.

        Args:
            df (pd.DataFrame): Input dataframe.
            home_or_visitor_suffix (str): Suffix to add to the streak column name.

        Returns:
            pd.DataFrame: Dataframe with added streak columns.
        """
        self.app_logger.structured_log(logging.INFO, "Calculating win/lose streaks -" + home_or_visitor_suffix)

        
        # Sort the dataframe by team and date
        df_sorted = df.sort_values([self.config.new_team_id_column, self.config.new_date_column])

        # Create a series of 1 for wins and -1 for losses
        win_loss_series = df_sorted[self.config.win_column].astype(int) * 2 - 1

        if self.config.extend_streaks:
            # Calculate streaks across all seasons
            streak = (
                win_loss_series.groupby((win_loss_series != win_loss_series.shift()).cumsum())
                .cumsum()
                .groupby(df_sorted[self.config.new_team_id_column])
                .shift()
                .fillna(0)
                .astype(int)
            )
        else:
            # Calculate streaks within each season
            streak = (
                win_loss_series.groupby([df_sorted[self.config.new_team_id_column], df_sorted[self.config.season_column]])
                .apply(lambda x: x.groupby((x != x.shift()).cumsum()).cumsum())
                .groupby(level=[0, 1])  # Group by team_id and season
                .shift()
                .fillna(0)
                .astype(int)
            )

        df[f'win_streak_{home_or_visitor_suffix}'] = streak

        self.app_logger.structured_log(
            logging.INFO,
            "Win/lose streaks calculated",
            output_shape=df.shape
        )

        return df


    def _merge_features(self, merged_df: pd.DataFrame, df_to_merge: pd.DataFrame, merge_columns: List[str], suffix: str) -> pd.DataFrame:
        """
        Merge features for the input dataframe.

        Args:
            merged_df (pd.DataFrame): Input dataframe.
            df_to_merge (pd.DataFrame): Input dataframe to merge.
            merge_columns (List[str]): Columns to merge.
            suffix (str): Suffix to add to the column name.

        Returns:    
            pd.DataFrame: Dataframe with merged features.
        """

        self.app_logger.structured_log(logging.INFO, "Merging features -" + suffix)

        
        columns_to_keep = [self.config.new_game_id_column, self.config.new_team_id_column] + merge_columns
        df_to_merge = df_to_merge[columns_to_keep]


        merged_df = merged_df.merge(
            df_to_merge,   
            how="left", 
            on=[self.config.new_game_id_column, self.config.new_team_id_column],
            suffixes=('', "_" + suffix) # suffixes should be handled already, but just in case
        )

        self.app_logger.structured_log(
            logging.INFO,
            "Features merged -" + suffix,
            output_shape=merged_df.shape
        )

        return merged_df



    def _calculate_home_visitor_streaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate home/visitor streaks for teams.
        Positive values indicate consecutive home games, negative values indicate consecutive visitor games.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with added home/visitor streak column.
        """
        self.app_logger.structured_log(logging.INFO, "Calculating home/visitor streaks")

        # Sort the dataframe by team and date
        df_sorted = df.sort_values([self.config.new_team_id_column, self.config.new_date_column])

        # Create a series of 1 for home games and -1 for visitor games
        home_visitor_series = df_sorted[self.config.home_team_column].astype(int) * 2 - 1

        if self.config.extend_streaks:
            # Calculate streaks across all seasons
            streak = (
                home_visitor_series.groupby((home_visitor_series != home_visitor_series.shift()).cumsum())
                .cumsum()
                .groupby(df_sorted[self.config.new_team_id_column])
                .shift()
                .fillna(0)
                .astype(int)
            )
        else:
            # Calculate streaks within each season
            streak = (
                home_visitor_series.groupby([df_sorted[self.config.new_team_id_column], df_sorted[self.config.season_column]])
                .apply(lambda x: x.groupby((x != x.shift()).cumsum()).cumsum())
                .groupby(level=[0, 1])  # Group by team_id and season
                .shift()
                .fillna(0)
                .astype(int)
            )

        df[f'home_visitor_streak'] = streak

        self.app_logger.structured_log(
            logging.INFO,
            "Home/visitor streaks calculated",
            output_shape=df.shape
        )

        return df
    
    @log_performance
    def merge_team_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge home (1 row) and visitor (1 row) team data for each game into a single row.
        This provides the model with all the info it needs to make a prediction on a single row.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with merged home and visitor team data.
        """
        self.app_logger.structured_log(
            logging.INFO,
            "Starting merge_team_data",
            dataframe_shape=df.shape
        )
        

        try:    
            # pull out just the game info columns - these will be the same for both home and visitor teams
            game_info_df = df[df['is_home_team'] == 1][self.config.game_info_columns].copy()
            
            # create a list of game info columns so we can drop these from the main home and visitor dataframes,
            # but keep the new_game_id_column so we can merge on it later
            game_info_columns = list(set(self.config.game_info_columns) - {self.config.new_game_id_column})
            
            df_without_game_info = df.drop(columns=game_info_columns)
            
            # Split into home and visitor dataframes
            df_home_team = df_without_game_info[df_without_game_info[self.config.home_team_column] == 1]
            df_visitor_team = df_without_game_info[df_without_game_info[self.config.home_team_column] == 0]

            # drop the post game stats columns (these are stats after the game is over and are not predictive)
            post_game_stats_columns = list(
                set(self.config.processed_schema) - 
                set(self.config.game_info_columns) - 
                set(self.config.team_info_columns) -
                {self.config.target_column}
            )
            
            df_home_team = df_home_team.drop(columns=post_game_stats_columns)
            df_visitor_team = df_visitor_team.drop(columns=post_game_stats_columns)
            df_visitor_team = df_visitor_team.drop(columns=self.config.target_column) #will only need one column indicating the game winner

            # add a prefix (like h_ or v_) to the columns of the home and visitor team dataframes
            df_home_team = df_home_team.add_prefix(self.config.home_team_prefix)
            # remove the "h_" prefix from the new_game_id_column so the column name is consistent for merging
            df_home_team =df_home_team.rename(columns={f'{self.config.home_team_prefix}{self.config.new_game_id_column}': self.config.new_game_id_column})
            
            # add a prefix to the columns of the visitor team dataframe
            df_visitor_team = df_visitor_team.add_prefix(self.config.visitor_team_prefix)
            # remove the "v_" prefix from the new_game_id_column so the column name is consistent for merging
            df_visitor_team =df_visitor_team.rename(columns={f'{self.config.visitor_team_prefix}{self.config.new_game_id_column}': self.config.new_game_id_column})
            
            # for each game, merge the home and visitor team data onto a single row
            merged_df = pd.merge(df_home_team, df_visitor_team, on=self.config.new_game_id_column, )
            
            # Merge back the game info columns
            final_df = pd.merge(game_info_df, merged_df, on=self.config.new_game_id_column, how='left')

            # move the new_game_id_column to the front
            final_df = final_df[[self.config.new_game_id_column] + [col for col in final_df.columns if col != self.config.new_game_id_column]]

            self.app_logger.structured_log(
                logging.INFO,
                "Team data combined successfully",
                dataframe_shape=final_df.shape
            )
            return final_df
        except Exception as e:
            raise FeatureEngineeringError ("Error in combine_team_data",
                                      self.app_logger,
                                      error_message=str(e),
                                      dataframe_shape=df.shape)
        

    @log_performance
    def encode_game_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode the game date into cyclic features and calculate season progress.

        Args:
            df (pd.DataFrame): Input dataframe containing game data.

        Returns:
            pd.DataFrame: Dataframe with encoded date features.

        Raises:
            FeatureEngineeringError: If there's an error during date encoding.
        """
        self.app_logger.structured_log(
            logging.INFO,
            "Starting game date encoding",
            input_shape=df.shape
        )

        try:
            # Convert date column to datetime if it's not already
            df[self.config.new_date_column] = pd.to_datetime(df[self.config.new_date_column])
            
            # Extract cyclic features
            df['day_of_week'] = df[self.config.new_date_column].dt.dayofweek
            df['month'] = df[self.config.new_date_column].dt.month
            
            # Encode day of week and month as cyclic features
            df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
            df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
            df['month_sin'] = np.sin((df['month'] - 1) * (2 * np.pi / 12))
            df['month_cos'] = np.cos((df['month'] - 1) * (2 * np.pi / 12))
            
            # Calculate season progress
            df['season_start'] = pd.to_datetime(df[self.config.season_column].astype(str) + '-' + self.config.season_start)
            df['season_end'] = pd.to_datetime((df[self.config.season_column] + 1).astype(str) + '-' + self.config.season_end)
            df['season_progress'] = (df[self.config.new_date_column] - df['season_start']) / (df['season_end'] - df['season_start'])
            
            # Drop intermediate columns
            df = df.drop(columns=['day_of_week', 'season_start', 'season_end']) # keep month for stratification

            # Update game_info_columns
            new_date_columns = ['month', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'season_progress']
            game_info_columns = self.config.game_info_columns.copy()
            game_info_columns.extend(new_date_columns)

            # Reorder columns
            df = df[game_info_columns + [col for col in df.columns if col not in game_info_columns]]

            self.app_logger.structured_log(
                logging.INFO,
                "Game date encoding completed",
                output_shape=df.shape,
                new_columns=new_date_columns
            )
        
            return df

        except Exception as e:
            raise FeatureEngineeringError("Error in game date encoding",
                                          self.app_logger,
                                          error_message=str(e),
                                          dataframe_shape=df.shape)
        
    @log_performance
    def _update_elo_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Update ELO ratings for teams based on game outcomes.

        IMPORTANT: Only pre-game ELO features are retained for modeling to prevent data leakage:
        - elo_team_before: Team's ELO rating before the game
        - elo_opp_before: Opponent's ELO rating before the game
        - elo_win_prob: Predicted win probability based on pre-game ELOs

        Post-game features (elo_change, elo_team_after) are calculated for ELO tracking
        but excluded from the final dataset to avoid leakage since they use game outcomes.

        Args:
            df (pd.DataFrame): Input dataframe containing game data.

        Returns:
            pd.DataFrame: Dataframe with pre-game ELO ratings and win probability.

        Raises:
            FeatureEngineeringError: If there's an error during ELO rating updates.
        """
        self.app_logger.structured_log(
            logging.INFO,
            "Starting ELO ratings update",
            input_shape=df.shape
        )

        try:
            # Ensure the DataFrame is sorted by date
            df = df.sort_values(self.config.new_date_column)

            # Initialize team ELO ratings
            team_elos = {}

            # Create new columns for ELO ratings and win probability
            # Note: We calculate all columns for ELO tracking, but will drop leaky ones later
            new_columns = [
                self.config.team_elo_before_column,
                self.config.opp_elo_before_column,
                self.config.win_prob_column,
                self.config.team_elo_after_column,
                self.config.elo_change_column
            ]

            # Check if we're updating only new rows or the entire dataframe
            if self.config.elo_update_new_rows_only:
                # Identify rows with missing ELO data
                mask = df[self.config.team_elo_before_column].isna()
                df.loc[mask, new_columns] = 0.0
                update_df = df[mask]
            else:
                df[new_columns] = 0.0
                update_df = df

            # Load existing ELO ratings if updating only new rows
            if self.config.elo_update_new_rows_only:
                last_known_elos = df.loc[~mask, [self.config.new_team_id_column, self.config.team_elo_after_column]]
                team_elos = dict(zip(last_known_elos[self.config.new_team_id_column], 
                                    last_known_elos[self.config.team_elo_after_column]))

            # Group the DataFrame by game_id
            grouped = update_df.groupby(self.config.new_game_id_column)

            # Iterate through each game


            self.app_logger.structured_log(
                logging.INFO,
                "Sample of home_team_column values",
                value_counts=str(df[self.config.home_team_column].value_counts())
            )

            for game_id, game in grouped:
                home_teams = game[game[self.config.home_team_column] == 1]
                away_teams = game[game[self.config.home_team_column] == 0]

                # Skip games that don't have both home and away teams
                if len(home_teams) == 0 or len(away_teams) == 0:
                    self.app_logger.structured_log(
                        logging.WARNING,
                        "Skipping game - missing home or away team",
                        game_id=game_id,
                        home_teams_count=len(home_teams),
                        away_teams_count=len(away_teams)
                    )
                    continue

                home_team = home_teams.iloc[0]
                away_team = away_teams.iloc[0]

                # Get or set initial ELO ratings
                home_elo = team_elos.get(home_team[self.config.new_team_id_column], self.config.initial_elo)
                away_elo = team_elos.get(away_team[self.config.new_team_id_column], self.config.initial_elo)
                
                # Adjust for home advantage
                home_elo_adj = home_elo + self.config.home_advantage
                
                # Calculate win probability for home team
                win_prob = 1 / (1 + 10 ** ((away_elo - home_elo_adj) / self.config.elo_width))
                
                # Determine actual outcome
                home_win = home_team[self.config.target_column]
                
                # Calculate ELO change
                elo_change = self.config.k_factor * (home_win - win_prob)
                
                # Update ELO ratings
                new_home_elo = home_elo + elo_change
                new_away_elo = away_elo - elo_change
                
                # Update DataFrame
                df.loc[home_team.name, self.config.team_elo_before_column] = home_elo
                df.loc[home_team.name, self.config.opp_elo_before_column] = away_elo
                df.loc[home_team.name, self.config.win_prob_column] = win_prob
                df.loc[home_team.name, self.config.team_elo_after_column] = new_home_elo
                df.loc[home_team.name, self.config.elo_change_column] = elo_change

                df.loc[away_team.name, self.config.team_elo_before_column] = away_elo
                df.loc[away_team.name, self.config.opp_elo_before_column] = home_elo
                df.loc[away_team.name, self.config.win_prob_column] = 1 - win_prob
                df.loc[away_team.name, self.config.team_elo_after_column] = new_away_elo
                df.loc[away_team.name, self.config.elo_change_column] = -elo_change
                
                # Update team ELO ratings for next game
                team_elos[home_team[self.config.new_team_id_column]] = new_home_elo
                team_elos[away_team[self.config.new_team_id_column]] = new_away_elo

            # IMPORTANT: Drop post-game ELO features to prevent data leakage
            # These features use the game outcome (target variable) in their calculation
            leaky_columns = [
                self.config.elo_change_column,  # Calculated from actual game result
                self.config.team_elo_after_column  # Includes elo_change (which uses target)
            ]

            columns_to_drop = [col for col in leaky_columns if col in df.columns]
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
                self.app_logger.structured_log(
                    logging.WARNING,
                    "Dropped post-game ELO features to prevent data leakage",
                    dropped_columns=columns_to_drop,
                    reason="These features are calculated using game outcomes (target variable)"
                )

            # Keep only pre-game features for modeling
            retained_elo_features = [
                self.config.team_elo_before_column,
                self.config.opp_elo_before_column,
                self.config.win_prob_column
            ]

            self.app_logger.structured_log(
                logging.INFO,
                "ELO ratings update completed",
                output_shape=df.shape,
                retained_elo_features=retained_elo_features
            )

            return df

        except Exception as e:
            raise FeatureEngineeringError("Error in updating ELO ratings",
                                            self.app_logger,
                                            error_message=str(e),
                                            game_id=game_id,
                                            dataframe_shape=df.shape)

    def apply_feature_allowlist(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataframe to only include allowlisted features.

        This should be called after all feature engineering is complete
        to ensure only the features from the allowlist are included in
        the final output.

        Args:
            df (pd.DataFrame): Dataframe with all engineered features

        Returns:
            pd.DataFrame: Dataframe with only allowlisted features (plus required columns)
        """
        if self.feature_allowlist is None:
            self.app_logger.structured_log(
                logging.INFO,
                "No feature allowlist enabled - returning all features"
            )
            return df

        # Always keep these columns regardless of allowlist
        # Note: After merge_team_data(), target column gets prefixed with h_
        target_col = self.config.target_column
        prefixed_target_col = f"{self.config.home_team_prefix}{target_col}"

        # Keep metadata columns that are useful for dashboard generation and debugging
        # These are explicitly excluded from model training via non_useful_columns config
        metadata_columns = [
            self.config.new_game_id_column,
            self.config.new_date_column,
            target_col if target_col in df.columns else prefixed_target_col,
            'h_team', 'v_team',           # Team abbreviations (e.g., "CHA", "OKC")
            'h_match_up', 'v_match_up'    # Match descriptions (e.g., "CHA vs. OKC")
        ]

        # Filter to only columns that actually exist in dataframe
        required_columns = [col for col in metadata_columns if col in df.columns]

        # Find columns to keep: required + allowlisted features
        columns_to_keep = []

        for col in df.columns:
            if col in required_columns:
                columns_to_keep.append(col)
            elif col in self.feature_allowlist:
                columns_to_keep.append(col)

        # Log filtering results
        metadata_kept = [c for c in columns_to_keep if c in required_columns]
        features_kept = [c for c in columns_to_keep if c not in required_columns]
        all_original_features = [c for c in df.columns if c not in required_columns]
        removed_features = set(all_original_features) - set(features_kept)

        self.app_logger.structured_log(
            logging.INFO,
            "Applied feature allowlist filter",
            metadata_columns_kept=len(metadata_kept),
            metadata_columns=metadata_kept,
            original_feature_count=len(all_original_features),
            kept_feature_count=len(features_kept),
            removed_feature_count=len(removed_features),
            allowlist_size=len(self.feature_allowlist),
            total_columns=len(columns_to_keep)
        )

        # Warn about allowlist features not found in dataframe
        missing_from_df = self.feature_allowlist - set(features_kept)
        if missing_from_df:
            self.app_logger.structured_log(
                logging.WARNING,
                "Some allowlist features not found in engineered data",
                missing_count=len(missing_from_df),
                sample_missing=list(missing_from_df)[:10]
            )

        return df[columns_to_keep]

    def _export_feature_schema(self, df: pd.DataFrame) -> None:
        """
        Export feature schema to JSON after feature engineering.

        This schema is consumed by ml_framework.preprocessing to determine
        which columns to scale, encode, etc.

        Args:
            df (pd.DataFrame): Dataframe with engineered features
        """
        try:
            self.app_logger.structured_log(
                logging.INFO,
                "Exporting feature schema",
                dataframe_shape=df.shape
            )

            # Determine columns to exclude from features
            exclude_columns = [
                self.config.new_game_id_column,
                self.config.new_team_id_column,
                self.config.new_date_column,
                self.config.target_column
            ]

            # Create schema from dataframe
            schema = FeatureSchema.from_dataframe(
                df,
                target_column=self.config.target_column,
                game_id_column=self.config.new_game_id_column,
                team_id_column=self.config.new_team_id_column,
                date_column=self.config.new_date_column,
                categorical_threshold=10,
                exclude_columns=exclude_columns
            )

            # Add feature groups for better organization
            schema.feature_groups = self._create_feature_groups(df)

            # Export to JSON
            schema_path = Path(self.config.processed_data_directory) / "feature_schema.json"
            schema.to_json(schema_path)

            self.app_logger.structured_log(
                logging.INFO,
                "Feature schema exported successfully",
                schema_path=str(schema_path),
                n_numeric=len(schema.numeric_features),
                n_categorical=len(schema.categorical_features),
                n_binary=len(schema.binary_features),
                n_total=len(schema.get_all_features())
            )

            # Log summary for reference
            self.app_logger.structured_log(
                logging.INFO,
                "Feature schema summary",
                summary=schema.summary()
            )

        except Exception as e:
            self.app_logger.structured_log(
                logging.WARNING,
                "Failed to export feature schema",
                error=str(e)
            )
            # Don't raise - schema export is helpful but not critical

    def _create_feature_groups(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Create logical groupings of features for better organization.

        Args:
            df (pd.DataFrame): Dataframe with engineered features

        Returns:
            Dictionary mapping group names to feature lists
        """
        feature_groups = {}

        # Group rolling average features by period and type
        for col in df.columns:
            if 'rolling_avg' in col:
                if 'rolling_avg_3_' in col:
                    feature_groups.setdefault('rolling_avg_3', []).append(col)
                elif 'rolling_avg_5_' in col:
                    feature_groups.setdefault('rolling_avg_5', []).append(col)
                elif 'rolling_avg_10_' in col:
                    feature_groups.setdefault('rolling_avg_10', []).append(col)

            # Group streak features
            elif 'streak' in col:
                feature_groups.setdefault('streaks', []).append(col)

            # Group ELO features (only pre-game features to prevent leakage)
            elif 'elo' in col.lower() or 'win_prob' in col:
                # Note: elo_change and elo_team_after are excluded to prevent data leakage
                feature_groups.setdefault('elo_ratings', []).append(col)

            # Group temporal features
            elif any(temporal in col for temporal in ['month', 'day_of_week', 'season_progress']):
                feature_groups.setdefault('temporal', []).append(col)

        return feature_groups

    @log_performance
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the dataframe into training and validation sets.
        The validation set will be selected from the most recent n completed seasons.
        (e.g. we will pull off the last n seasons, split off a portion for validation, then add the remainder back to the original dataframe)

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and validation dataframes.
        """
        self.app_logger.structured_log(logging.INFO, "Splitting data into training and validation sets")
        try:
            # Get season column from config
            season_column = self.config.season_column

            # determine the last season in the dataframe
            last_season = df[season_column].max()

            # determine the start season for the validation set
            validation_start_season = last_season - self.config.validation_last_n_seasons

            # limit the working dataframe to the last n seasons
            working_df = df[df[season_column] >= validation_start_season]
            df = df.drop(working_df.index)

            # use a stratified split to ensure that seasonality is maintained in the validation set (e.g. same number of games from each month)
            training_df, validation_df = train_test_split(
                working_df,
                test_size=self.config.validation_split,
                stratify=working_df[self.config.stratify_column],
                random_state=self.config.random_state
            )

            training_df = pd.concat([df, training_df])

            self.app_logger.structured_log(logging.INFO, "Data split completed",
                           training_shape=training_df.shape,
                           validation_shape=validation_df.shape)

            return training_df, validation_df
        except Exception as e:
            raise FeatureEngineeringError("Error in splitting data",
                                        self.app_logger,
                                        error_message=str(e),
                                        dataframe_shape=df.shape)