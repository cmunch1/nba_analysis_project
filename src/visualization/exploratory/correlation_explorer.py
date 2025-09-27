import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import numpy as np
from .base_explorer import BaseExplorer

class CorrelationExplorer(BaseExplorer):
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

    def scatter_plot(self, x: str, y: str, hue: Optional[str] = None) -> None:
        """Create a scatter plot of two variables, with optional color coding."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x, y=y, hue=hue, data=self.df)
        plt.title(f'Scatter Plot: {x} vs {y}')
        plt.show() 