import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Any, Optional
import shap
from sklearn.inspection import partial_dependence
from .base_chart import BaseChart

class FeatureCharts(BaseChart):
    def create_feature_importance_chart(self, feature_importance: np.ndarray, 
                                      feature_names: List[str], 
                                      top_n: int = 20) -> plt.Figure:
        """
        Create a feature importance chart.

        Args:
            feature_importance (np.ndarray): Array of feature importance scores
            feature_names (List[str]): List of feature names
            top_n (int): Number of top features to display

        Returns:
            plt.Figure: Feature importance chart
        """
        try:
            # Sort features by importance and get top_n features
            indices = np.argsort(np.abs(feature_importance))[-top_n:]
            top_importance = feature_importance[indices]
            top_names = [feature_names[i] for i in indices]

            # Create figure and plot
            fig, ax = self._create_figure()
            y_pos = np.arange(len(top_importance))
            ax.barh(y_pos, np.abs(top_importance))
            
            # Customize the plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_names)
            ax.invert_yaxis()
            ax.set_xlabel('Feature Importance (absolute value)')
            ax.grid(True, axis='x', linestyle='--', alpha=0.7)
            
            return self._finalize_plot(fig, f'Top {top_n} Most Important Features')
            
        except Exception as e:
            self._handle_error(e, "feature importance chart",
                             feature_importance_shape=feature_importance.shape,
                             feature_names_length=len(feature_names))

    def create_partial_dependence_plot(self, model: Any, 
                                     X: pd.DataFrame, 
                                     feature_names: List[str], 
                                     target_names: Optional[List[str]] = None) -> plt.Figure:
        """
        Create a partial dependence plot for specified features.
        """
        try:
            pdp_results = partial_dependence(
                model, X, features=feature_names, kind="average", grid_resolution=20
            )

            fig, axes = plt.subplots(nrows=1, ncols=len(feature_names), 
                                   figsize=(6*len(feature_names), 5))
            if len(feature_names) == 1:
                axes = [axes]

            for i, (ax, feature_name) in enumerate(zip(axes, feature_names)):
                pdp_values = pdp_results["average"][i]
                feature_values = pdp_results["values"][i]
                
                ax.plot(feature_values, pdp_values.T)
                ax.set_xlabel(feature_name)
                ax.set_ylabel('Partial dependence')
                
                if target_names and len(target_names) > 1:
                    ax.legend(target_names, loc='best')

            return self._finalize_plot(fig, 'Partial Dependence Plot')
            
        except Exception as e:
            self._handle_error(e, "partial dependence plot",
                             model_type=type(model).__name__,
                             dataframe_shape=X.shape,
                             feature_names=feature_names) 