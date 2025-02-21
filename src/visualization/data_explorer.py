from .exploratory.base_explorer import BaseExplorer
from .exploratory.correlation_explorer import CorrelationExplorer
from .exploratory.distribution_explorer import DistributionExplorer
from .exploratory.time_series_explorer import TimeSeriesExplorer
from .exploratory.team_performance_explorer import TeamPerformanceExplorer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
import numpy as np

class DataExplorer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.base = BaseExplorer(df)
        self.correlation = CorrelationExplorer(df)
        self.distribution = DistributionExplorer(df)
        self.time_series = TimeSeriesExplorer(df)
        self.team = TeamPerformanceExplorer(df)
        
    def basic_stats(self) -> pd.DataFrame:
        """Calculate basic statistics for numerical columns."""
        return self.base.basic_stats()

    def missing_values(self) -> pd.DataFrame:
        """Check for missing values in the dataset."""
        return self.base.missing_values()

    def correlation_matrix(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Generate a correlation matrix for specified columns or all numeric columns."""
        return self.correlation.correlation_matrix(columns)

    def plot_correlation_heatmap(self, columns: Optional[List[str]] = None) -> None:
        """Plot a heatmap of the correlation matrix."""
        self.correlation.plot_correlation_heatmap(columns)

    def distribution_plot(self, column: str) -> None:
        """Plot the distribution of a specific column."""
        self.distribution.distribution_plot(column)

    def boxplot(self, x: str, y: str) -> None:
        """Create a boxplot to compare distributions."""
        self.distribution.boxplot(x, y)

    def scatter_plot(self, x: str, y: str, hue: Optional[str] = None) -> None:
        """Create a scatter plot of two variables, with optional color coding."""
        self.correlation.scatter_plot(x, y, hue)

    def time_series_plot(self, date_column: str, value_column: str) -> None:
        """Plot a time series of a specific column."""
        self.time_series.time_series_plot(date_column, value_column)

    def group_analysis(self, group_by: str, agg_column: str, agg_func: str = 'mean') -> pd.DataFrame:
        """Perform groupby analysis on the dataset."""
        return self.df.groupby(group_by)[agg_column].agg(agg_func).sort_values(ascending=False)

    def win_loss_distribution(self, team_column: str, result_column: str) -> pd.DataFrame:
        """Analyze win-loss distribution for teams."""
        return self.team.win_loss_distribution(team_column, result_column)

    def top_performers(self, team_column: str, metric_column: str, n: int = 10) -> pd.DataFrame:
        """Identify top performing teams based on a specific metric."""
        return self.team.top_performers(team_column, metric_column, n)

    def home_away_performance(self, team_column: str, location_column: str, metric_column: str) -> pd.DataFrame:
        """Compare team performance in home vs away games."""
        return self.team.home_away_performance(team_column, location_column, metric_column)

    def streak_analysis(self, team_column: str, date_column: str, result_column: str) -> pd.DataFrame:
        """Analyze winning and losing streaks for teams."""
        return self.team.streak_analysis(team_column, date_column, result_column)

    def performance_trend(self, team_column: str, date_column: str, metric_column: str, window: int = 5) -> pd.DataFrame:
        """Calculate moving average of a performance metric for teams."""
        return self.team.performance_trend(team_column, date_column, metric_column, window)