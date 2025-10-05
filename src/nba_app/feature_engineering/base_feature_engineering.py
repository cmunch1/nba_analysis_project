from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple

class AbstractFeatureEngineer(ABC):
    @abstractmethod
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to engineer features.
        
        Args:
            df (pd.DataFrame): The input dataframe.
        
        Returns:
            pd.DataFrame: Dataframe with engineered features.
        """
        pass

    @abstractmethod
    def merge_team_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to merge home and away team data for each game into a single row.
        
        Args:
            df (pd.DataFrame): The input dataframe.
        
        Returns:
            pd.DataFrame: Dataframe with merged home and away team data.
        """
        pass



class AbstractFeatureSelector(ABC):
    @abstractmethod
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass
