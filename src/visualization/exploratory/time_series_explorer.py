import matplotlib.pyplot as plt
from .base_explorer import BaseExplorer

class TimeSeriesExplorer(BaseExplorer):
    def time_series_plot(self, date_column: str, value_column: str) -> None:
        """Plot a time series of a specific column."""
        df_sorted = self.df.sort_values(date_column)
        plt.figure(figsize=(12, 6))
        plt.plot(df_sorted[date_column], df_sorted[value_column])
        plt.title(f'Time Series: {value_column} over time')
        plt.xlabel(date_column)
        plt.ylabel(value_column)
        plt.xticks(rotation=45)
        plt.show() 