from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Any



class AbstractNBADataProcessor(ABC):
    @abstractmethod
    def process_data(self, data: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Abstract method to process NBA data.
        
        Args:
            data (List[pd.DataFrame]): The data to process.
        
        Returns:
            pd.DataFrame: Processed data as a single DataFrame.
        """
        pass


