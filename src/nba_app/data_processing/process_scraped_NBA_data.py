import pandas as pd
import logging
from typing import List, Tuple, Dict
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging import log_performance, structured_log
from ml_framework.core.error_handling.error_handler import DataProcessingError
from .base_data_processing_classes import BaseNBADataProcessor

logger = logging.getLogger(__name__)

class ProcessScrapedNBAData(BaseNBADataProcessor):
    @log_performance
    def __init__(self,  config: BaseConfigManager):
        """
        Initialize the ProcessScrapedNBAData class.

        Args:
            config (BaseConfigManager): Configuration object containing processing parameters.
        """
        self.config = config
        
        structured_log(logger, logging.INFO, "ProcessScrapedNBAData initialized",
                       config_type=type(config).__name__)
        
    @log_performance
    def process_data(self, scraped_dataframes: List[pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Process the scraped NBA data through all steps.

        Args:
            scraped_dataframes (List[pd.DataFrame]): List of scraped dataframes.

        Returns:
            pd.DataFrame: Fully processed dataframe.

        Raises:
            DataProcessingError: If there's an error during any step of the data processing.
        """
        
        structured_log(logger, logging.INFO, "Starting process_data",
                       dataframe_count=len(scraped_dataframes))
        try:
                        
            df = self._merge_dataframes(scraped_dataframes)
            df = self._handle_anomalous_data(df)
            df, column_mapping = self._transform_data(df)

            df[self.config.new_date_column] = pd.to_datetime(df[self.config.new_date_column])
            df = df.sort_values(by=[self.config.new_date_column, self.config.new_game_id_column, self.config.home_team_column], ascending=[True, True, True])
                      
            structured_log(logger, logging.INFO, "Data processing completed successfully",
                           final_dataframe_shape=df.shape)
            return df, column_mapping
        
        except (DataProcessingError) as e:
            structured_log(logger, logging.ERROR, "Error in process_data",
                           error_message=str(e),
                           error_type=type(e).__name__)
            raise
        except Exception as e:
            structured_log(logger, logging.ERROR, "Unexpected error in process_data",
                           error_message=str(e),
                           error_type=type(e).__name__)
            raise DataProcessingError("Unexpected error in process_data",
                                      error_message=str(e))
        
    @log_performance
    def merge_team_data(self, df: pd.DataFrame) -> pd.DataFrame:
        structured_log(logger, logging.INFO, "Starting merge_team_data",
                       dataframe_shape=df.shape)
        try:    
            # separate out the game info columns but keep the new_game_id_column so we can merge on it
            game_info_columns = [col for col in self.config.game_info_columns if col != self.config.new_game_id_column]
            game_info_df = df[df['is_home_team'] == 1][game_info_columns + [self.config.new_game_id_column]].copy()
            
            # Remove game info columns from the main dataframe, but keep new_game_id_column
            df_without_game_info = df.drop(columns=game_info_columns)
            
            # Split into home and away dataframes
            df_home_team = df_without_game_info[df_without_game_info[self.config.home_team_column] == 1]
            df_away_team = df_without_game_info[df_without_game_info[self.config.home_team_column] == 0]
            
            # Merge home and away data
            merged_df = pd.merge(df_home_team, df_away_team, on=self.config.new_game_id_column, suffixes=("_" + self.config.home_game_suffix, "_" + self.config.visitor_game_suffix))
            
            # Merge back the game info columns
            final_df = pd.merge(game_info_df, merged_df, on=self.config.new_game_id_column)

            # move the new_game_id_column to the front
            final_df = final_df[[self.config.new_game_id_column] + [col for col in final_df.columns if col != self.config.new_game_id_column]]

            structured_log(logger, logging.INFO, "Team data combined successfully",
                           dataframe_shape=final_df.shape)
            return final_df
        except Exception as e:
            raise DataProcessingError("Error in combine_team_data",
                                      error_message=str(e),
                                      dataframe_shape=df.shape)
        


    @log_performance
    def _merge_dataframes(self, scraped_dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge all the different stat type dataframes into a single dataframe.

        Args:
            scraped_dataframes (List[pd.DataFrame]): List of dataframes to merge.

        Returns:
            pd.DataFrame: Merged dataframe.

        Raises:
            DataProcessingError: If there's an error during the merging process.
        """
        structured_log(logger, logging.INFO, "Starting merge_dataframes",
                       dataframe_count=len(scraped_dataframes))
        try:

            merged = None
            for i, df in enumerate(scraped_dataframes):
                df.columns = df.columns.str.replace('\xa0', ' ') # Replace non-breaking space with regular space
                df = df.rename(columns={'TEAM': 'Team', 'MATCH UP': 'Match Up', 'GAME DATE': 'Game Date'})
                if i == 0:
                    merged = df
                else:
                    merged = pd.merge(merged, df, on=[self.config.game_id_column, self.config.team_id_column], suffixes=('', '_dupe'))

            merged = merged.drop(columns=merged.filter(regex='_dupe').columns)
            structured_log(logger, logging.INFO, "Dataframes merged successfully",
                           merged_shape=merged.shape)    
                                                 
            return merged
        
        except Exception as e:
            raise DataProcessingError("Error in merge_dataframes",
                                      error_message=str(e),
                                      dataframe_count=len(scraped_dataframes))

    @log_performance
    def _handle_anomalous_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle occurrences of anomalous data. The data from the NBA website is very clean, but there are still some anomalies.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Processed dataframe with missing data handled.

        Raises:
            DataProcessingError: If there's an error during missing data handling.
        """
        structured_log(logger, logging.INFO, "Starting handle_anomalous_data",
                       initial_nan_count=int(df.isna().sum().sum()))  
        try:
            # Drop rows where 'W/L' is NaN. These games were not played. Normally these are not included in the data, but there was a case where this happened.
            df = df.dropna(subset=['W/L'])

            # A team once had no free throw attempts, and the free throw percentage was set as "-" (for 0/0) in a column that is normally numeric.
            # The following code makes sure that any time something like this happens, the free throw percentage is set to a value designated in the config.
            # This code also handles the also unlikely cases if no field goals were attempted or if no 3 pointers were attempted.
            for col in df.columns:
                if '%' in col:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(self.config.default_percentage_value)

            # check that all the nans are gone
            remaining_nans = int(df.isna().sum().sum())
            structured_log(logger, logging.INFO, "Anomalous data handling complete",
                           remaining_nan_count=remaining_nans)
            return df
        except Exception as e:
            raise DataProcessingError("Error in handle_anomalous_data",
                                      error_message=str(e),
                                      dataframe_shape=df.shape)

    @log_performance
    def _transform_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Transform data by renaming columns, extracting new columns, and reordering columns.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Transformed dataframe.

        Raises:
            DataProcessingError: If there's an error during data transformation.
        """
        structured_log(logger, logging.INFO, "Starting transform_data",
                       initial_column_count=len(df.columns))
        try:
            df, column_mapping = self._rename_columns(df)
            df = self._extract_new_columns(df)
            df = self._reorder_columns(df)

            
            structured_log(logger, logging.INFO, "Data transformation complete",
                           final_column_count=len(df.columns))
            return df, column_mapping
        except Exception as e:
            raise DataProcessingError("Error in transform_data",
                                      error_message=str(e),
                                      dataframe_shape=df.shape)



    @log_performance
    def _rename_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Rename columns of the dataframe for consistency and to remove special characters and spaces.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            Tuple[pd.DataFrame, Dict[str, str]]: Processed dataframe and column mapping dictionary.

        Raises:
            DataProcessingError: If there's an error during column renaming.
        """
        structured_log(logger, logging.INFO, "Starting rename_columns",
                       original_column_count=len(df.columns))
        try:
            original_columns = df.columns.tolist()

            df.columns = df.columns.str.lower().str.replace(' ', '_')
            df.columns = df.columns.str.replace('%', '_pct_')
            df.columns = df.columns.str.replace('rtg', '_rtg')  
            df.columns = df.columns.str.replace('__', '_')
            df.columns = df.columns.str.rstrip('_')
            df.columns = df.columns.str.lstrip('_')
            df.columns = df.columns.str.replace('+/-', 'plus_minus')
            df.columns = df.columns.str.replace('ast/to', 'ast_turnover_ratio')

            column_mapping = dict(zip(original_columns, df.columns.tolist()))
            structured_log(logger, logging.INFO, "Columns renamed successfully",
                           renamed_column_count=len(column_mapping))
            return df, column_mapping
        except Exception as e:
            raise DataProcessingError("Error in rename_columns",
                                      error_message=str(e),
                                      dataframe_shape=df.shape)

    @log_performance
    def _extract_new_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract new columns from existing data, such as flagging the home team, overtime games, and playoffs.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with new columns added.

        Raises:
            DataProcessingError: If there's an error during new column extraction.
        """
        initial_column_count=len(df.columns)
        structured_log(logger, logging.INFO, "Starting extract_new_columns",
                       initial_column_count=initial_column_count)
        try:
            # note that a previous step converted column names to lowercase and substituted an underscore for a space
            df[self.config.win_column] = df["w/l"].str.contains("W").astype(int)
            df = df.drop(columns=['w/l'])

            df[self.config.home_team_column] = df["match_up"].str.contains("vs.").astype(int)
            df[self.config.is_overtime_column] = (df["min"] > self.config.regular_game_minutes).astype(int)

            df[self.config.new_game_id_column] = df[self.config.new_game_id_column].astype(str)
            df[self.config.season_column] = df[self.config.new_game_id_column].str[1:3].astype(int) + self.config.season_year_offset
            df[self.config.sub_season_id_column] = df[self.config.new_game_id_column].str[:1].astype(int)
            df[self.config.new_game_id_column] = df[self.config.new_game_id_column].str[1:]
            df[self.config.is_playoff_column] = (df[self.config.sub_season_id_column] > self.config.regular_season_game_id_threshold).astype(int)
            
            

            new_column_count = len(df.columns) - initial_column_count
            structured_log(logger, logging.INFO, "New columns extracted successfully",
                           new_column_count=new_column_count)
            return df
        except Exception as e:
            raise DataProcessingError("Error in extract_new_columns",
                                      error_message=str(e),
                                      dataframe_shape=df.shape)

    @log_performance
    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder columns of the dataframe so that the columns are more logically ordered.
         - All game info (game id, date, is playoff, etc ...) is grouped together and first, 
         - followed by the team info (team id, team name, is home team, etc...), 
         - and then the team stats are grouped together.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with reordered columns.

        Raises:
            DataProcessingError: If there's an error during column reordering.
        """
        structured_log(logger, logging.INFO, "Starting reorder_columns",
                       initial_column_count=len(df.columns))
        try:
            all_columns = df.columns.tolist()

            game_info = self.config.game_info_columns
            team_info = self.config.team_info_columns
            team_stats = [col for col in all_columns if col not in game_info + team_info]

            reordered_df = df[game_info + team_info + team_stats]
            structured_log(logger, logging.INFO, "Columns reordered successfully",
                           reordered_column_count=len(reordered_df.columns))
            return reordered_df
        except Exception as e:
            raise DataProcessingError("Error in reorder_columns",
                                      error_message=str(e),
                                      dataframe_shape=df.shape)



