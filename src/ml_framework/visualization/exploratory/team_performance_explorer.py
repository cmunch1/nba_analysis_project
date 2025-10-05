import pandas as pd
from .base_explorer import BaseExplorer

class TeamPerformanceExplorer(BaseExplorer):
    def win_loss_distribution(self, team_column: str, result_column: str) -> pd.DataFrame:
        """
        Analyze win-loss distribution for teams.
        
        Args:
            team_column (str): Column name containing team identifiers
            result_column (str): Column name containing win/loss results
            
        Returns:
            pd.DataFrame: Win-loss distribution for each team
        """
        return self.df.groupby(team_column)[result_column].value_counts(normalize=True).unstack()

    def top_performers(self, team_column: str, metric_column: str, n: int = 10) -> pd.DataFrame:
        """
        Identify top performing teams based on a specific metric.
        
        Args:
            team_column (str): Column name containing team identifiers
            metric_column (str): Column name containing performance metric
            n (int): Number of top teams to return
            
        Returns:
            pd.DataFrame: Top n teams sorted by metric
        """
        return self.df.groupby(team_column)[metric_column].mean().sort_values(ascending=False).head(n)

    def home_away_performance(self, team_column: str, location_column: str, metric_column: str) -> pd.DataFrame:
        """
        Compare team performance in home vs away games.
        
        Args:
            team_column (str): Column name containing team identifiers
            location_column (str): Column name containing location (home/away)
            metric_column (str): Column name containing performance metric
            
        Returns:
            pd.DataFrame: Home vs away performance for each team
        """
        return self.df.groupby([team_column, location_column])[metric_column].mean().unstack()

    def streak_analysis(self, team_column: str, date_column: str, result_column: str) -> pd.DataFrame:
        """
        Analyze winning and losing streaks for teams.
        
        Args:
            team_column (str): Column name containing team identifiers
            date_column (str): Column name containing dates
            result_column (str): Column name containing win/loss results
            
        Returns:
            pd.DataFrame: Maximum streak lengths for each team
        """
        df_sorted = self.df.sort_values([team_column, date_column])
        df_sorted['streak'] = (df_sorted.groupby(team_column)[result_column]
                             .transform(lambda x: x.ne(x.shift()).cumsum()))
        return df_sorted.groupby([team_column, result_column, 'streak']).size().unstack(level=1).max()

    def performance_trend(self, team_column: str, date_column: str, metric_column: str, window: int = 5) -> pd.DataFrame:
        """
        Calculate moving average of a performance metric for teams.
        
        Args:
            team_column (str): Column name containing team identifiers
            date_column (str): Column name containing dates
            metric_column (str): Column name containing performance metric
            window (int): Window size for moving average
            
        Returns:
            pd.DataFrame: Rolling average performance for each team
        """
        df_sorted = self.df.sort_values([team_column, date_column])
        return df_sorted.groupby(team_column)[metric_column].rolling(window=window).mean().reset_index() 