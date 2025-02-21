import matplotlib.pyplot as plt
import seaborn as sns
from .base_explorer import BaseExplorer

class DistributionExplorer(BaseExplorer):
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