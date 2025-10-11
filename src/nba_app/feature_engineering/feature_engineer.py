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
from typing import List, Dict, Optional
from pathlib import Path
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

        self.app_logger.structured_log(
            logging.INFO,
            "FeatureEngineer initialized",
            config_type=type(config).__name__
        )

    @staticmethod
    def log_performance(func):
        """Decorator factory for performance logging"""
        def wrapper(*args, **kwargs):
            # Get the self instance from args since this is now a static method
            instance = args[0]
            return instance.app_logger.log_performance(func)(*args, **kwargs)
        return wrapper

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

            self.app_logger.structured_log(
                logging.INFO,
                "Feature engineering completed",
                output_shape=df_merged.shape
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

        Returns:
            pd.DataFrame: Dataframe with rolling averages.
        """
        self.app_logger.structured_log(logging.INFO, "Creating rolling averages -" + home_or_visitor_suffix)
        all_columns = set(df.columns)
        stats_columns = all_columns.difference(self.config.game_info_columns + self.config.team_info_columns)
        periods = self.config.team_rolling_avg_periods
        
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

        Args:
            df (pd.DataFrame): Input dataframe containing game data.

        Returns:
            pd.DataFrame: Dataframe with updated ELO ratings and related columns.

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
            
            self.app_logger.structured_log(
                logging.INFO,
                "ELO ratings update completed",
                output_shape=df.shape,
                new_columns=new_columns
            )
            
            return df

        except Exception as e:
            raise FeatureEngineeringError("Error in updating ELO ratings",
                                            self.app_logger,
                                            error_message=str(e),
                                            game_id=game_id,
                                            dataframe_shape=df.shape)

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

            # Group ELO features
            elif 'elo' in col.lower() or 'win_prob' in col:
                feature_groups.setdefault('elo_ratings', []).append(col)

            # Group temporal features
            elif any(temporal in col for temporal in ['month', 'day_of_week', 'season_progress']):
                feature_groups.setdefault('temporal', []).append(col)

        return feature_groups