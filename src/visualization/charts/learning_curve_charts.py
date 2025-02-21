import matplotlib.pyplot as plt
from typing import Dict, Any
from .base_chart import BaseChart
from ...common.data_classes import ModelTrainingResults

class LearningCurveCharts(BaseChart):
    def create_learning_curve(self, results: ModelTrainingResults) -> plt.Figure:
        """
        Create learning curve showing average performance across folds.
        
        Args:
            results: ModelTrainingResults containing averaged learning curve data
            
        Returns:
            plt.Figure: Learning curve visualization
        """
        try:
            plot_data = results.learning_curve_data.get_plot_data()
            if not plot_data:
                return None

            fig, ax = self._create_figure(figsize=(10, 6))
            
            # Plot averaged training and validation scores
            ax.plot(plot_data['iterations'], plot_data['train_scores'], 
                   label='Training score', color='blue', alpha=0.8)
            ax.plot(plot_data['iterations'], plot_data['val_scores'], 
                   label='Validation score', color='orange', alpha=0.8)
            
            # Add labels and customize
            ax.set_xlabel('Iteration')
            ax.set_ylabel(f'Score ({results.learning_curve_data.metric_name})')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='best')
            
            title = f'Learning Curve - {results.model_name}\nAveraged across {results.n_folds} folds'
            return self._finalize_plot(fig, title)
            
        except Exception as e:
            self._handle_error(e, "learning curve",
                             model_name=results.model_name,
                             n_folds=results.n_folds) 