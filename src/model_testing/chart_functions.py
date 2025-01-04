import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from ..logging.logging_utils import log_performance, structured_log
from ..error_handling.custom_exceptions import ChartCreationError
from sklearn.inspection import partial_dependence

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
    def create_shap_summary_plot(self, model: Any, X: pd.DataFrame) -> plt.Figure:
        """
        Create a SHAP summary plot.

        Args:
            model (Any): Trained model object.
            X (pd.DataFrame): Feature dataframe.

        Returns:
            plt.Figure: Matplotlib figure object containing the SHAP summary plot.
        """
        structured_log(logger, logging.INFO, "Creating SHAP summary plot", input_shape=X.shape)
        try:
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            plt.title("SHAP Summary Plot")

            structured_log(logger, logging.INFO, "SHAP summary plot created successfully")
            return fig
        except Exception as e:
            raise ChartCreationError("Error creating SHAP summary plot",
                                     error_message=str(e),
                                     model_type=type(model).__name__,
                                     dataframe_shape=X.shape)

    @log_performance
    def create_shap_force_plot(self, model: Any, X: pd.DataFrame, index: int = 0) -> plt.Figure:
        """
        Create a SHAP force plot for a single observation.

        Args:
            model (Any): Trained model object.
            X (pd.DataFrame): Feature dataframe.
            index (int): Index of the observation to explain. Defaults to 0.

        Returns:
            plt.Figure: Matplotlib figure object containing the SHAP force plot.
        """
        structured_log(logger, logging.INFO, "Creating SHAP force plot", input_shape=X.shape, index=index)
        try:
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X.iloc[index])

            # Create the plot
            fig, ax = plt.subplots(figsize=(16, 3))
            shap.force_plot(explainer.expected_value, shap_values, X.iloc[index], matplotlib=True, show=False, ax=ax)
            plt.title(f"SHAP Force Plot for Observation {index}")

            structured_log(logger, logging.INFO, "SHAP force plot created successfully")
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
            cm = confusion_matrix(y_true, y_pred)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title('Confusion Matrix')
            
            structured_log(logger, logging.INFO, "Confusion matrix created successfully")
            return fig
        except Exception as e:
            raise ChartCreationError("Error creating confusion matrix",
                                     error_message=str(e),
                                     y_true_shape=y_true.shape,
                                     y_pred_shape=y_pred.shape)

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
            fpr, tpr, _ = roc_curve(y_true, y_score)
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
            
            structured_log(logger, logging.INFO, "ROC curve created successfully")
            return fig
        except Exception as e:
            raise ChartCreationError("Error creating ROC curve",
                                     error_message=str(e),
                                     y_true_shape=y_true.shape,
                                     y_score_shape=y_score.shape)

    @log_performance
    def create_learning_curve(self, model: Any, X: np.ndarray, y: np.ndarray) -> plt.Figure:
        """
        Create a learning curve to evaluate model performance with varying training set sizes.

        Args:
            model (Any): Scikit-learn compatible model object.
            X (np.ndarray): Feature array.
            y (np.ndarray): Target array.

        Returns:
            plt.Figure: Matplotlib figure object containing the learning curve.
        """
        structured_log(logger, logging.INFO, "Creating learning curve")
        try:
            from sklearn.model_selection import learning_curve
            
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
            ax.plot(train_sizes, test_mean, label='Cross-validation score', color='green', marker='s')
            ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
            ax.set_xlabel('Number of training examples')
            ax.set_ylabel('Score')
            ax.set_title('Learning Curve')
            ax.legend(loc='lower right')
            
            structured_log(logger, logging.INFO, "Learning curve created successfully")
            return fig
        except Exception as e:
            raise ChartCreationError("Error creating learning curve",
                                     error_message=str(e),
                                     model_type=type(model).__name__,
                                     X_shape=X.shape,
                                     y_shape=y.shape)

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
    def create_shap_dependence_plot(self, shap_values: shap.Explanation, features: pd.DataFrame, feature_name: str, interaction_feature: str = None) -> plt.Figure:
        """
        Create a SHAP dependence plot for a specific feature.

        Args:
            shap_values (shap.Explanation): SHAP values calculated for the model.
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
                ax=ax
            )
            ax.set_title(f'SHAP Dependence Plot for {feature_name}')
            
            structured_log(logger, logging.INFO, "SHAP dependence plot created successfully")
            return fig
        except Exception as e:
            raise ChartCreationError("Error creating SHAP dependence plot",
                                     error_message=str(e),
                                     feature_name=feature_name,
                                     interaction_feature=interaction_feature)
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
    def generate_charts(
        self,
        model: Any,
        X: pd.DataFrame,
        feature_names: List[str],
        feature_importance: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        y_score: Optional[np.ndarray] = None,
        shap_values: Optional[Any] = None,
        target_names: Optional[List[str]] = None,
    ) -> Dict[str, plt.Figure]:
        """
        Generate all configured charts.
        
        Args:
            model: Trained model object
            X: Feature dataframe
            feature_names: List of feature names
            feature_importance: Array of feature importance scores
            y_true: True labels for classification metrics
            y_pred: Predicted labels
            y_score: Predicted probabilities/scores
            shap_values: Pre-calculated SHAP values (optional)
            target_names: List of target names for multi-output models
            
        Returns:
            Dict[str, plt.Figure]: Dictionary mapping chart names to matplotlib figures
        """
        structured_log(logger, logging.INFO, "Starting chart generation")
        charts = {}

        try:
            # Feature Importance Chart
            if getattr(self.config, 'feature_importance_chart', False) and feature_importance is not None:
                structured_log(logger, logging.INFO, "Generating feature importance chart")
                charts['feature_importance'] = self.chart_functions.create_feature_importance_chart(
                    feature_importance=feature_importance,
                    feature_names=feature_names
                )

            # SHAP Summary Plot
            if getattr(self.config, 'shap_summary_plot', False):
                structured_log(logger, logging.INFO, "Generating SHAP summary plot")
                charts['shap_summary'] = self.chart_functions.create_shap_summary_plot(
                    model=model,
                    X=X
                )

            # SHAP Force Plot
            if getattr(self.config, 'shap_force_plot', False):
                structured_log(logger, logging.INFO, "Generating SHAP force plot")
                charts['shap_force'] = self.chart_functions.create_shap_force_plot(
                    model=model,
                    X=X
                )

            # SHAP Dependence Plot
            if getattr(self.config, 'shap_dependence_plot', False) and shap_values is not None:
                structured_log(logger, logging.INFO, "Generating SHAP dependence plots")
                for feature in feature_names[:min(3, len(feature_names))]:  # Limit to top 3 features
                    charts[f'shap_dependence_{feature}'] = self.chart_functions.create_shap_dependence_plot(
                        shap_values=shap_values,
                        features=X,
                        feature_name=feature
                    )

            # Confusion Matrix (if classification metrics are provided)
            if y_true is not None and y_pred is not None:
                structured_log(logger, logging.INFO, "Generating confusion matrix")
                charts['confusion_matrix'] = self.chart_functions.create_confusion_matrix(
                    y_true=y_true,
                    y_pred=y_pred
                )

            # ROC Curve (if classification scores are provided)
            if y_true is not None and y_score is not None:
                structured_log(logger, logging.INFO, "Generating ROC curve")
                charts['roc_curve'] = self.chart_functions.create_roc_curve(
                    y_true=y_true,
                    y_score=y_score
                )

            # Learning Curve
            if isinstance(X, np.ndarray) and y_true is not None:
                structured_log(logger, logging.INFO, "Generating learning curve")
                charts['learning_curve'] = self.chart_functions.create_learning_curve(
                    model=model,
                    X=X,
                    y=y_true
                )

            # Partial Dependence Plot
            if len(feature_names) > 0:
                structured_log(logger, logging.INFO, "Generating partial dependence plots")
                selected_features = feature_names[:min(3, len(feature_names))]  # Limit to top 3 features
                charts['partial_dependence'] = self.chart_functions.create_partial_dependence_plot(
                    model=model,
                    X=X,
                    feature_names=selected_features,
                    target_names=target_names
                )

            structured_log(logger, logging.INFO, f"Chart generation completed. Generated {len(charts)} charts")
            return charts

        except Exception as e:
            raise ChartCreationError("Error in chart generation orchestration",
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




