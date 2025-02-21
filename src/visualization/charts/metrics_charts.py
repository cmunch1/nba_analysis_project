import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from typing import Optional
from .base_chart import BaseChart

class MetricsCharts(BaseChart):
    def create_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
        """
        Create a confusion matrix for classification models.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels

        Returns:
            plt.Figure: Confusion matrix visualization
        """
        try:
            # Filter out NaN values
            mask = ~np.isnan(y_pred)
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            # Check if we have enough samples after filtering
            if len(y_pred_clean) < 2:
                raise ValueError("Insufficient non-NaN samples for confusion matrix calculation")
            
            cm = confusion_matrix(y_true_clean, y_pred_clean)
            
            fig, ax = self._create_figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            
            return self._finalize_plot(fig, 'Confusion Matrix')
            
        except Exception as e:
            self._handle_error(e, "confusion matrix",
                             y_true_shape=y_true.shape,
                             y_pred_shape=y_pred.shape,
                             nan_count=np.sum(np.isnan(y_pred)))

    def create_roc_curve(self, y_true: np.ndarray, y_score: np.ndarray) -> plt.Figure:
        """
        Create a ROC curve for binary classification models.

        Args:
            y_true (np.ndarray): True labels
            y_score (np.ndarray): Predicted probabilities or scores

        Returns:
            plt.Figure: ROC curve visualization
        """
        try:
            # Filter out NaN values
            mask = ~np.isnan(y_score)
            y_true_clean = y_true[mask]
            y_score_clean = y_score[mask]
            
            # Check if we have enough samples after filtering
            if len(y_score_clean) < 2:
                raise ValueError("Insufficient non-NaN samples for ROC curve calculation")
            
            fpr, tpr, _ = roc_curve(y_true_clean, y_score_clean)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = self._create_figure(figsize=(10, 8))
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")
            
            return self._finalize_plot(fig, 'Receiver Operating Characteristic (ROC) Curve')
            
        except Exception as e:
            self._handle_error(e, "ROC curve",
                             y_true_shape=y_true.shape,
                             y_score_shape=y_score.shape,
                             nan_count=np.sum(np.isnan(y_score))) 