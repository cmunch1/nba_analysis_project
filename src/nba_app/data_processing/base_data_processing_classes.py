from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Any



class BaseNBADataProcessor(ABC):
    @abstractmethod
    def process_data(self, data: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Base method to process NBA data.

        Args:
            data (List[pd.DataFrame]): The data to process.

        Returns:
            pd.DataFrame: Processed data as a single DataFrame.
        """
        pass

    @abstractmethod
    def merge_team_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Base method to merge team data.

        Args:
            df (pd.DataFrame): The dataframe to merge.

        Returns:
            pd.DataFrame: Merged dataframe.
        """
        pass