import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class BaseExplorer:
    def __init__(self, df: pd.DataFrame):
        """Initialize base explorer with dataframe."""
        self.df = df
        
    def basic_stats(self) -> pd.DataFrame:
        """Calculate basic statistics for numerical columns."""
        return self.df.describe()

    def missing_values(self) -> pd.DataFrame:
        """Check for missing values in the dataset."""
        missing = self.df.isnull().sum()
        missing_percent = 100 * missing / len(self.df)
        return pd.DataFrame({
            'Missing Values': missing, 
            'Percent Missing': missing_percent
        }) 