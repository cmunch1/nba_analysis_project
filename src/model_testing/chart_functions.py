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
