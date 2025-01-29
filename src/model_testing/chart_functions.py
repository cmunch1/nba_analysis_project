import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from ..logging.logging_utils import log_performance, structured_log
from ..error_handling.custom_exceptions import ChartCreationError
from sklearn.inspection import partial_dependence
from .data_classes import ModelTrainingResults

logger = logging.getLogger(__name__)

class ChartFunctions:
    @log_performance
    def __init__(self):
        """
        Initialize the ChartFunctions class.
        """
        structured_log(logger, logging.INFO, "ChartFunctions initialized")

    @log_performance
    def create_feature_importance_chart(self, feature_importance: np.ndarray, feature_names: List[str], top_n: int = 20) -> plt.Figure:
        """
        Create a feature importance chart.

        Args:
            feature_importance (np.ndarray): Array of feature importance scores.
            feature_names (List[str]): List of feature names.
            top_n (int): Number of top features to display. Defaults to 20.

        Returns:
            plt.Figure: Matplotlib figure object containing the feature importance chart.
        """
        structured_log(logger, logging.INFO, "Creating feature importance chart", top_n=top_n)
        try:
            # Sort features by importance
            indices = np.argsort(feature_importance)[::-1]
            top_features = indices[:top_n]

            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x=feature_importance[top_features], y=[feature_names[i] for i in top_features], ax=ax)
            ax.set_title(f"Top {top_n} Feature Importances")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Features")

            structured_log(logger, logging.INFO, "Feature importance chart created successfully")
            return fig
        except Exception as e:
            raise ChartCreationError("Error creating feature importance chart",
                                     error_message=str(e),
                                     feature_importance_shape=feature_importance.shape,
                                     feature_names_length=len(feature_names))

    @log_performance
    def create_shap_summary_plot(self, model: Any, X: pd.DataFrame, shap_values: Optional[np.ndarray] = None) -> plt.Figure:
        """
        Create a SHAP summary plot.

        Args:
            model (Any): Trained model object.
            X (pd.DataFrame): Feature dataframe.
            shap_values (Optional[np.ndarray]): Pre-calculated SHAP values.

        Returns:
            plt.Figure: Matplotlib figure object containing the SHAP summary plot.
        """
        structured_log(logger, logging.INFO, "Creating SHAP summary plot", input_shape=X.shape)
        try:
            # Use pre-calculated SHAP values if available
            if shap_values is None:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)

            # Create the plot
            fig = plt.figure(figsize=(12, 8))  # Store the figure object
            values = shap_values[1] if isinstance(shap_values, list) else shap_values
            shap.summary_plot(values, X, plot_type="bar", show=False)
            plt.title("SHAP Summary Plot")

            structured_log(logger, logging.INFO, "SHAP summary plot created successfully")
            return fig
        except Exception as e:
            raise ChartCreationError("Error creating SHAP summary plot",
                                   error_message=str(e),
                                   model_type=type(model).__name__,
                                   dataframe_shape=X.shape)

    @log_performance
    def create_shap_force_plot(self, model: Any, X: pd.DataFrame, index: int = 0, shap_values: Optional[np.ndarray] = None) -> plt.Figure:
        """
        Create a SHAP force plot for a single observation.

        Args:
            model (Any): Trained model object.
            X (pd.DataFrame): Feature dataframe.
            index (int): Index of the observation to explain. Defaults to 0.
            shap_values (Optional[np.ndarray]): Pre-calculated SHAP values.

        Returns:
            plt.Figure: Matplotlib figure object containing the SHAP force plot.
        """
        structured_log(logger, logging.INFO, "Creating SHAP force plot", input_shape=X.shape, index=index)
        try:
            # Use pre-calculated SHAP values if available
            if shap_values is not None:
                values = shap_values[index]
                expected_value = values.sum() / 2  # For binary classification
            else:
                # Calculate SHAP values if not provided
                explainer = shap.TreeExplainer(model)
                X_sample = X.iloc[[index]]
                shap_values = explainer.shap_values(X_sample)
                values = shap_values[1] if isinstance(shap_values, list) else shap_values
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

            # Create figure and plot
            plt.figure(figsize=(16, 3))
            X_sample = X.iloc[[index]] if isinstance(X.iloc[index], pd.Series) else X.iloc[index]
            shap.force_plot(
                expected_value,
                values,
                X_sample,
                matplotlib=True,
                show=False
            )
            fig = plt.gcf()
            plt.title(f"SHAP Force Plot for Observation {index}")
            return fig
        except Exception as e:
            raise ChartCreationError("Error creating SHAP force plot",
                                   error_message=str(e),
                                   model_type=type(model).__name__,
                                   dataframe_shape=X.shape,
                                   index=index)

    @log_performance
    def create_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
        """
        Create a confusion matrix for classification models.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            plt.Figure: Matplotlib figure object containing the confusion matrix.
        """
        structured_log(logger, logging.INFO, "Creating confusion matrix")
        try:
            from sklearn.metrics import confusion_matrix
            
            # Filter out NaN values
            mask = ~np.isnan(y_pred)
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            # Log NaN filtering results
            nan_count = np.sum(~mask)
            if nan_count > 0:
                structured_log(logger, logging.WARNING, 
                             "Filtered NaN values from predictions",
                             total_samples=len(y_pred),
                             nan_count=nan_count,
                             remaining_samples=len(y_pred_clean))
            
            # Check if we have enough samples after filtering
            if len(y_pred_clean) < 2:
                raise ValueError("Insufficient non-NaN samples for confusion matrix calculation")
            
            cm = confusion_matrix(y_true_clean, y_pred_clean)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title('Confusion Matrix')
            
            structured_log(logger, logging.INFO, "Confusion matrix created successfully",
                          nan_filtered=nan_count)
            return fig
        except Exception as e:
            raise ChartCreationError("Error creating confusion matrix",
                                     error_message=str(e),
                                     y_true_shape=y_true.shape,
                                     y_pred_shape=y_pred.shape,
                                     nan_count=np.sum(np.isnan(y_pred)))

    @log_performance
    def create_roc_curve(self, y_true: np.ndarray, y_score: np.ndarray) -> plt.Figure:
        """
        Create a ROC curve for binary classification models.

        Args:
            y_true (np.ndarray): True labels.
            y_score (np.ndarray): Predicted probabilities or scores.

        Returns:
            plt.Figure: Matplotlib figure object containing the ROC curve.
        """
        structured_log(logger, logging.INFO, "Creating ROC curve")
        try:
            from sklearn.metrics import roc_curve, auc
            
            # Filter out NaN values
            mask = ~np.isnan(y_score)
            y_true_clean = y_true[mask]
            y_score_clean = y_score[mask]
            
            # Log NaN filtering results
            nan_count = np.sum(~mask)
            if nan_count > 0:
                structured_log(logger, logging.WARNING, 
                             "Filtered NaN values from predictions",
                             total_samples=len(y_score),
                             nan_count=nan_count,
                             remaining_samples=len(y_score_clean))
            
            # Check if we have enough samples after filtering
            if len(y_score_clean) < 2:
                raise ValueError("Insufficient non-NaN samples for ROC curve calculation")
            
            fpr, tpr, _ = roc_curve(y_true_clean, y_score_clean)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax.legend(loc="lower right")
            
            structured_log(logger, logging.INFO, "ROC curve created successfully",
                          auc_score=roc_auc)
            return fig
        except Exception as e:
            raise ChartCreationError("Error creating ROC curve",
                                   error_message=str(e),
                                   y_true_shape=y_true.shape,
                                   y_score_shape=y_score.shape,
                                   nan_count=np.sum(np.isnan(y_score)))


    @log_performance
    def create_learning_curve(self, results: ModelTrainingResults) -> plt.Figure:
        """
        Create a learning curve from stored training results.
        
        Args:
            results (ModelTrainingResults): Results object containing learning curve data
            
        Returns:
            plt.Figure: Learning curve visualization
        """
        structured_log(logger, logging.INFO, "Creating learning curve from results")
        try:
            # Verify learning curve data exists and is properly aggregated
            if not hasattr(results, 'learning_curve_data') or not results.learning_curve_data:
                raise ValueError("No learning curve data available in results")

            # Access the aggregated data
            aggregated_data = results.learning_curve_data.get('aggregated')
            if not aggregated_data:
                raise ValueError("No aggregated learning curve data available")

            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot mean scores with error bands
            train_sizes = aggregated_data['train_sizes']
            train_scores_mean = aggregated_data['train_scores_mean']
            train_scores_std = aggregated_data['train_scores_std']
            val_scores_mean = aggregated_data['val_scores_mean']
            val_scores_std = aggregated_data['val_scores_std']

            # Plot training scores
            ax.plot(train_sizes, train_scores_mean, 'o-', label='Training score',
                    color='blue')
            ax.fill_between(train_sizes, 
                           train_scores_mean - train_scores_std,
                           train_scores_mean + train_scores_std, 
                           alpha=0.1, color='blue')

            # Plot validation scores
            ax.plot(train_sizes, val_scores_mean, 'o-', label='Cross-validation score',
                    color='orange')
            ax.fill_between(train_sizes, 
                           val_scores_mean - val_scores_std,
                           val_scores_mean + val_scores_std, 
                           alpha=0.1, color='orange')

            ax.set_xlabel('Training examples')
            ax.set_ylabel('Score')
            ax.set_title(f'Learning Curve ({results.model_name})')
            ax.grid(True)
            ax.legend(loc='best')
            
            structured_log(logger, logging.INFO, "Learning curve created successfully")
            return fig
            
        except Exception as e:
            raise ChartCreationError("Error creating learning curve",
                                error_message=str(e))
    
    @log_performance
    def create_partial_dependence_plot(self, model: Any, X: pd.DataFrame, feature_names: List[str], target_names: List[str] = None) -> plt.Figure:
        """
        Create a partial dependence plot for specified features.

        Args:
            model (Any): Trained model object.
            X (pd.DataFrame): Feature dataframe.
            feature_names (List[str]): List of feature names to plot.
            target_names (List[str], optional): List of target names for multi-output models.

        Returns:
            plt.Figure: Matplotlib figure object containing the partial dependence plot.
        """
        structured_log(logger, logging.INFO, "Creating partial dependence plot", features=feature_names)
        try:
            # Calculate partial dependence
            pdp_results = partial_dependence(
                model, X, features=feature_names, kind="average", grid_resolution=20
            )

            # Create the plot
            fig, axes = plt.subplots(nrows=1, ncols=len(feature_names), figsize=(6*len(feature_names), 5))
            if len(feature_names) == 1:
                axes = [axes]  # Ensure axes is always a list

            for i, (ax, feature_name) in enumerate(zip(axes, feature_names)):
                pdp_values = pdp_results["average"][i]
                feature_values = pdp_results["values"][i]
                
                ax.plot(feature_values, pdp_values.T)
                ax.set_xlabel(feature_name)
                ax.set_ylabel('Partial dependence')
                
                if target_names and len(target_names) > 1:
                    ax.legend(target_names, loc='best')

            plt.tight_layout()
            fig.suptitle('Partial Dependence Plot', fontsize=16)
            plt.subplots_adjust(top=0.9)

            structured_log(logger, logging.INFO, "Partial dependence plot created successfully")
            return fig
        except Exception as e:
            raise ChartCreationError("Error creating partial dependence plot",
                                     error_message=str(e),
                                     model_type=type(model).__name__,
                                     dataframe_shape=X.shape,
                                     feature_names=feature_names)

    @log_performance
    def create_shap_dependence_plot(self, shap_values: np.ndarray, features: pd.DataFrame, feature_name: str, interaction_feature: str = None) -> plt.Figure:
        """
        Create a SHAP dependence plot for a specific feature.

        Args:
            shap_values (np.ndarray): SHAP values calculated for the model.
            features (pd.DataFrame): Feature dataframe used for predictions.
            feature_name (str): Name of the main feature to plot.
            interaction_feature (str, optional): Name of the feature to use for coloring (to show interactions).

        Returns:
            plt.Figure: Matplotlib figure object containing the SHAP dependence plot.
        """
        structured_log(logger, logging.INFO, "Creating SHAP dependence plot", feature=feature_name, interaction_feature=interaction_feature)
        try:
            fig, ax = plt.subplots(figsize=(10, 7))
            shap.dependence_plot(
                feature_name, 
                shap_values, 
                features, 
                interaction_index=interaction_feature,
                ax=ax,
                show=False
            )
            ax.set_title(f'SHAP Dependence Plot for {feature_name}')
            
            structured_log(logger, logging.INFO, "SHAP dependence plot created successfully")
            return fig
        except Exception as e:
            raise ChartCreationError("Error creating SHAP dependence plot",
                                     error_message=str(e),
                                     feature_name=feature_name,
                                     interaction_feature=interaction_feature)

    @log_performance
    def create_shap_waterfall_plot(self, model: Any, X: pd.DataFrame, index: int = 0, shap_values: Optional[np.ndarray] = None) -> plt.Figure:
        """
        Create a SHAP waterfall plot for a single observation.

        Args:
            model (Any): Trained model object.
            X (pd.DataFrame): Feature dataframe.
            index (int): Index of the observation to explain. Defaults to 0.
            shap_values (Optional[np.ndarray]): Pre-calculated SHAP values.

        Returns:
            plt.Figure: Matplotlib figure object containing the SHAP waterfall plot.
        """
        structured_log(logger, logging.INFO, "Creating SHAP waterfall plot", input_shape=X.shape, index=index)
        try:
            # Use pre-calculated SHAP values if available
            if shap_values is not None:
                values = shap_values[index]
                expected_value = values.sum() / 2  # For binary classification
            else:
                # Calculate SHAP values if not provided
                explainer = shap.TreeExplainer(model)
                X_sample = X.iloc[[index]]
                shap_values = explainer.shap_values(X_sample)
                values = shap_values[1] if isinstance(shap_values, list) else shap_values
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

            # Create figure and plot
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=values.reshape(-1),
                    base_values=expected_value,
                    data=X.iloc[index].values,
                    feature_names=X.columns
                ),
                show=False
            )
            fig = plt.gcf()
            plt.title(f"SHAP Waterfall Plot for Observation {index}")
            return fig
        except Exception as e:
            raise ChartCreationError("Error creating SHAP waterfall plot",
                                   error_message=str(e),
                                   model_type=type(model).__name__,
                                   dataframe_shape=X.shape,
                                   index=index)

    @log_performance
    def create_shap_beeswarm_plot(self, model: Any, X: pd.DataFrame, shap_values: Optional[np.ndarray] = None) -> plt.Figure:
        """
        Create a SHAP beeswarm plot.

        Args:
            model (Any): Trained model object.
            X (pd.DataFrame): Feature dataframe.
            shap_values (Optional[np.ndarray]): Pre-calculated SHAP values.

        Returns:
            plt.Figure: Matplotlib figure object containing the SHAP beeswarm plot.
        """
        structured_log(logger, logging.INFO, "Creating SHAP beeswarm plot", input_shape=X.shape)
        try:
            # Use pre-calculated SHAP values if available
            if shap_values is None:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)

            # Create figure and plot
            plt.figure(figsize=(12, 8))
            values = shap_values[1] if isinstance(shap_values, list) else shap_values
            shap.summary_plot(values, X, plot_type="dot", show=False)
            fig = plt.gcf()
            plt.title("SHAP Beeswarm Plot")
            return fig
        except Exception as e:
            raise ChartCreationError("Error creating SHAP beeswarm plot",
                                   error_message=str(e),
                                   model_type=type(model).__name__,
                                   dataframe_shape=X.shape)

class ChartOrchestrator:
    @log_performance
    def __init__(self, config):
        """
        Initialize the ChartOrchestrator with configuration.
        
        Args:
            config: Configuration object containing chart flags
        """
        self.chart_functions = ChartFunctions()  # Create instance internally
        self.config = config
        structured_log(logger, logging.INFO, "ChartOrchestrator initialized")

    @log_performance
    def create_charts_from_results(self, results: ModelTrainingResults) -> Dict[str, plt.Figure]:
        """
        Create all available charts from a ModelTrainingResults object.
        
        Args:
            results: ModelTrainingResults object containing model evaluation data
            
        Returns:
            Dict[str, plt.Figure]: Dictionary mapping chart names to matplotlib figures
        """
        structured_log(logger, logging.INFO, "Creating charts from ModelTrainingResults")
        charts = {}
        
        try:
            # Feature Importance Chart
            if results.feature_importance_scores is not None:
                charts['feature_importance'] = self.create_feature_importance_chart(
                    feature_importance=results.feature_importance_scores,
                    feature_names=results.feature_names
                )

            # Confusion Matrix
            if results.binary_predictions is not None:
                charts['confusion_matrix'] = self.create_confusion_matrix(
                    y_true=results.target_data,
                    y_pred=results.binary_predictions
                )

            # ROC Curve
            if results.probability_predictions is not None:
                charts['roc_curve'] = self.create_roc_curve(
                    y_true=results.target_data,
                    y_score=results.probability_predictions
                )

            # SHAP Summary Plot
            if results.model is not None and results.feature_data is not None:
                charts['shap_summary'] = self.create_shap_summary_plot(
                    model=results.model,
                    X=results.feature_data
                )

            # Learning Curve
            if all(x is not None for x in [results.model, results.feature_data, results.target_data]):
                charts['learning_curve'] = self.create_learning_curve(
                    results=results
                )

            structured_log(logger, logging.INFO, f"Created {len(charts)} charts successfully")
            return charts

        except Exception as e:
            raise ChartCreationError("Error creating charts from results",
                                   error_message=str(e))

    @log_performance
    def generate_charts(
        self,
        results: ModelTrainingResults
    ) -> Dict[str, plt.Figure]:
        """
        Generate all configured charts based on ModelTrainingResults.
        
        Args:
            results: ModelTrainingResults object containing all necessary data
            
        Returns:
            Dict[str, plt.Figure]: Dictionary mapping chart names to matplotlib figures
        """
        structured_log(logger, logging.INFO, "Starting chart generation")
        charts = {}

        try:
            chart_data = results.prepare_for_charting()

            # Feature Importance Chart
            if getattr(self.config, 'feature_importance_chart', False):
                try:    
                    if chart_data["feature_importance"] is not None:
                        charts['feature_importance'] = self.chart_functions.create_feature_importance_chart(
                            feature_importance=chart_data["feature_importance"],
                            feature_names=chart_data["feature_names"]
                        )
                except Exception as e:
                    structured_log(logger, logging.WARNING, 
                                 "Failed to create feature importance chart",
                                 error=str(e))

            # Confusion Matrix
            if getattr(self.config, 'confusion_matrix', False):
                try:
                    if chart_data["y_true"] is not None and chart_data["y_pred"] is not None:
                        charts['confusion_matrix'] = self.chart_functions.create_confusion_matrix(
                            y_true=chart_data["y_true"],
                            y_pred=chart_data["y_pred"]
                        )
                except Exception as e:
                    structured_log(logger, logging.WARNING, 
                                 "Failed to create confusion matrix",
                                 error=str(e))

            # ROC Curve
            if getattr(self.config, 'roc_curve', False):
                try:
                    if chart_data["y_true"] is not None and chart_data["y_prob"] is not None:
                        charts['roc_curve'] = self.chart_functions.create_roc_curve(
                            y_true=chart_data["y_true"],
                            y_score=chart_data["y_prob"]
                        )
                except Exception as e:
                    structured_log(logger, logging.WARNING, 
                                 "Failed to create ROC curve",
                                 error=str(e))

            # SHAP Summary Plot
            if getattr(self.config, 'shap_summary_plot', False):
                try:
                    if chart_data["model"] is not None and chart_data["X"] is not None:
                        charts['shap_summary'] = self.chart_functions.create_shap_summary_plot(
                            model=chart_data["model"],
                            X=chart_data["X"],
                            shap_values=results.shap_values
                        )
                except Exception as e:
                    structured_log(logger, logging.WARNING, 
                                 "Failed to create SHAP summary plot",
                                 error=str(e))

            # SHAP Force Plot
            if getattr(self.config, 'shap_force_plot', False):
                try:
                    if chart_data["model"] is not None and chart_data["X"] is not None:
                        charts['shap_force'] = self.chart_functions.create_shap_force_plot(
                            model=chart_data["model"],
                            X=chart_data["X"],
                            shap_values=results.shap_values
                        )
                except Exception as e:
                    structured_log(logger, logging.WARNING, 
                                 "Failed to create SHAP force plot",
                                 error=str(e))

            # SHAP Dependence Plot
            if getattr(self.config, 'shap_dependence_plot', False):
                try:
                    if chart_data["X"] is not None and results.shap_values is not None:
                        charts['shap_dependence'] = self.chart_functions.create_shap_dependence_plot(
                            shap_values=results.shap_values,
                            features=chart_data["X"],
                            feature_name=chart_data["feature_names"][0]  # Using first feature as example
                        )
                except Exception as e:
                    structured_log(logger, logging.WARNING, 
                                 "Failed to create SHAP dependence plot",
                                 error=str(e))

            # Learning Curve
            if getattr(self.config, 'learning_curve', False):
                try:
                    charts['learning_curve'] = self.chart_functions.create_learning_curve(
                        results=results
                    )
                except Exception as e:
                    structured_log(logger, logging.WARNING, 
                                 "Failed to create learning curve",
                                 error=str(e))

            # SHAP Waterfall Plot
            if getattr(self.config, 'shap_waterfall_plot', False):
                try:
                    if chart_data["model"] is not None and chart_data["X"] is not None:
                        charts['shap_waterfall'] = self.chart_functions.create_shap_waterfall_plot(
                            model=chart_data["model"],
                            X=chart_data["X"],
                            shap_values=results.shap_values
                        )
                except Exception as e:
                    structured_log(logger, logging.WARNING, 
                                 "Failed to create SHAP waterfall plot",
                                 error=str(e))

            # SHAP Beeswarm Plot
            if getattr(self.config, 'shap_beeswarm_plot', False):
                try:
                    if chart_data["model"] is not None and chart_data["X"] is not None:
                        charts['shap_beeswarm'] = self.chart_functions.create_shap_beeswarm_plot(
                            model=chart_data["model"],
                            X=chart_data["X"],
                            shap_values=results.shap_values
                        )
                except Exception as e:
                    structured_log(logger, logging.WARNING, 
                                 "Failed to create SHAP beeswarm plot",
                                 error=str(e))

            structured_log(logger, logging.INFO, f"Chart generation completed. Generated {len(charts)} charts")
            return charts

        except Exception as e:
            raise ChartCreationError("Error in chart generation",
                                   error_message=str(e),
                                   charts_generated=list(charts.keys()))

    @log_performance
    def save_charts(self, charts: Dict[str, plt.Figure], output_dir: str) -> None:
        """
        Save generated charts to files.
        
        Args:
            charts: Dictionary of chart names and their corresponding matplotlib figures
            output_dir: Directory to save the charts
        """
        import os
        structured_log(logger, logging.INFO, "Saving charts", output_dir=output_dir)
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            for name, fig in charts.items():
                output_path = os.path.join(output_dir, f"{name}.png")
                fig.savefig(output_path, bbox_inches='tight', dpi=300)
                plt.close(fig)  # Clean up memory
                
            structured_log(logger, logging.INFO, "Charts saved successfully", 
                         chart_count=len(charts), 
                         output_dir=output_dir)
                         
        except Exception as e:
            raise ChartCreationError("Error saving charts",
                                   error_message=str(e),
                                   output_dir=output_dir)




