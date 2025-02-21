import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any, Optional, List
import shap
from .base_chart import BaseChart

class SHAPCharts(BaseChart):
    def create_shap_summary_plot(self, model: Any, X: pd.DataFrame, 
                               shap_values: Optional[np.ndarray] = None, 
                               n_features: Optional[int] = None) -> plt.Figure:
        """
        Create a SHAP summary plot.

        Args:
            model: Trained model object
            X: Feature dataframe
            shap_values: Pre-calculated SHAP values
            n_features: Number of top features to display

        Returns:
            plt.Figure: SHAP summary plot
        """
        try:
            # Calculate SHAP values if not provided
            if shap_values is None:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                values = shap_values[1]  # For binary classification
            else:
                values = shap_values

            # Ensure values is 2D
            if values.ndim == 1:
                values = values.reshape(-1, 1)

            # Handle NaN values
            if np.any(np.isnan(values)):
                valid_mask = ~np.any(np.isnan(values), axis=1)
                values = values[valid_mask]
                X = X.iloc[valid_mask]

            # Limit features if specified
            if n_features is not None:
                feature_importance = np.abs(values).mean(0)
                top_indices = np.argsort(feature_importance)[-n_features:]
                values = values[:, top_indices]
                X = X.iloc[:, top_indices]

            fig = plt.figure(figsize=(12, 8))
            shap.summary_plot(values, X, plot_type="bar", show=False)
            
            return self._finalize_plot(fig, "SHAP Summary Plot")
            
        except Exception as e:
            self._handle_error(e, "SHAP summary plot",
                             model_type=type(model).__name__,
                             dataframe_shape=X.shape)

    def create_shap_dependence_plot(self, shap_values: np.ndarray, 
                                  features: pd.DataFrame, 
                                  feature_name: str, 
                                  interaction_feature: Optional[str] = None) -> plt.Figure:
        """
        Create a SHAP dependence plot for a specific feature.

        Args:
            shap_values: SHAP values
            features: Feature dataframe
            feature_name: Name of the main feature to plot
            interaction_feature: Name of the feature to use for coloring

        Returns:
            plt.Figure: SHAP dependence plot
        """
        try:
            fig = plt.figure(figsize=(10, 7))
            shap.dependence_plot(
                feature_name, 
                shap_values, 
                features, 
                interaction_index=interaction_feature,
                ax=plt.gca(),
                show=False
            )
            
            return self._finalize_plot(fig, f'SHAP Dependence Plot for {feature_name}')
            
        except Exception as e:
            self._handle_error(e, "SHAP dependence plot",
                             feature_name=feature_name,
                             interaction_feature=interaction_feature)

    def create_shap_waterfall_plot(self, model: Any, X: pd.DataFrame, 
                                 index: int = 0, 
                                 shap_values: Optional[np.ndarray] = None) -> plt.Figure:
        """
        Create a SHAP waterfall plot for a single observation.

        Args:
            model: Trained model object
            X: Feature dataframe
            index: Index of the observation to explain
            shap_values: Pre-calculated SHAP values

        Returns:
            plt.Figure: SHAP waterfall plot
        """
        try:
            # Calculate SHAP values if not provided
            if shap_values is not None:
                values = shap_values[index]
                expected_value = values.sum() / 2
            else:
                explainer = shap.TreeExplainer(model)
                X_sample = X.iloc[[index]]
                shap_values = explainer.shap_values(X_sample)
                values = shap_values[1] if isinstance(shap_values, list) else shap_values
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

            fig = plt.figure(figsize=(16, 3))
            X_sample = X.iloc[[index]] if isinstance(X.iloc[index], pd.Series) else X.iloc[index]
            shap.waterfall_plot(
                shap.Explanation(
                    values=values.reshape(-1),
                    base_values=expected_value,
                    data=X_sample.values,
                    feature_names=X.columns
                ),
                show=False
            )
            
            return self._finalize_plot(fig, f'SHAP Waterfall Plot for Observation {index}')
            
        except Exception as e:
            self._handle_error(e, "SHAP waterfall plot",
                             model_type=type(model).__name__,
                             dataframe_shape=X.shape,
                             index=index) 