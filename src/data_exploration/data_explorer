import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
import numpy as np

class DataExplorer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def basic_stats(self) -> pd.DataFrame:
        """Calculate basic statistics for numerical columns."""
        return self.df.describe()

    def missing_values(self) -> pd.DataFrame:
        """Check for missing values in the dataset."""
        missing = self.df.isnull().sum()
        missing_percent = 100 * missing / len(self.df)
        return pd.DataFrame({'Missing Values': missing, 'Percent Missing': missing_percent})

    def correlation_matrix(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Generate a correlation matrix for specified columns or all numeric columns."""
        if columns:
            return self.df[columns].corr()
        return self.df.select_dtypes(include=[np.number]).corr()

    def plot_correlation_heatmap(self, columns: Optional[List[str]] = None) -> None:
        """Plot a heatmap of the correlation matrix."""
        corr = self.correlation_matrix(columns)
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.show()

    def distribution_plot(self, column: str) -> None:
        """Plot the distribution of a specific column."""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    def boxplot(self, x: str, y: str) -> None:
        """Create a boxplot to compare distributions."""
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=x, y=y, data=self.df)
        plt.title(f'Boxplot of {y} by {x}')
        plt.xticks(rotation=45)
        plt.show()

    def scatter_plot(self, x: str, y: str, hue: Optional[str] = None) -> None:
        """Create a scatter plot of two variables, with optional color coding."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x, y=y, hue=hue, data=self.df)
        plt.title(f'Scatter Plot: {x} vs {y}')
        plt.show()

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

    def group_analysis(self, group_by: str, agg_column: str, agg_func: str = 'mean') -> pd.DataFrame:
        """Perform groupby analysis on the dataset."""
        return self.df.groupby(group_by)[agg_column].agg(agg_func).sort_values(ascending=False)

    def win_loss_distribution(self, team_column: str, result_column: str) -> pd.DataFrame:
        """Analyze win-loss distribution for teams."""
        return self.df.groupby(team_column)[result_column].value_counts(normalize=True).unstack()

    def top_performers(self, team_column: str, metric_column: str, n: int = 10) -> pd.DataFrame:
        """Identify top performing teams based on a specific metric."""
        return self.df.groupby(team_column)[metric_column].mean().sort_values(ascending=False).head(n)

    def home_away_performance(self, team_column: str, location_column: str, metric_column: str) -> pd.DataFrame:
        """Compare team performance in home vs away games."""
        return self.df.groupby([team_column, location_column])[metric_column].mean().unstack()

    def streak_analysis(self, team_column: str, date_column: str, result_column: str) -> pd.DataFrame:
        """Analyze winning and losing streaks for teams."""
        df_sorted = self.df.sort_values([team_column, date_column])
        df_sorted['streak'] = (df_sorted.groupby(team_column)[result_column]
                               .transform(lambda x: x.ne(x.shift()).cumsum()))
        return df_sorted.groupby([team_column, result_column, 'streak']).size().unstack(level=1).max()

    def performance_trend(self, team_column: str, date_column: str, metric_column: str, window: int = 5) -> pd.DataFrame:
        """Calculate moving average of a performance metric for teams."""
        df_sorted = self.df.sort_values([team_column, date_column])
        return df_sorted.groupby(team_column)[metric_column].rolling(window=window).mean().reset_index()