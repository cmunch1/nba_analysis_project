"""Data classes for model metrics and learning curves."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

@dataclass
class LearningCurvePoint:
    """Data point for learning curves."""
    train_size: int
    train_score: float
    val_score: float
    fold: int
    iteration: Optional[int] = None
    metric_name: Optional[str] = None

@dataclass
class LearningCurveData:
    """Container for learning curve data averaged across folds."""
    train_scores: List[float] = field(default_factory=list)
    val_scores: List[float] = field(default_factory=list)
    iterations: List[int] = field(default_factory=list)
    metric_name: Optional[str] = None

    def add_iteration(self, train_score: float, val_score: float, iteration: int):
        """Add scores for a single iteration."""
        self.train_scores.append(train_score)
        self.val_scores.append(val_score)
        self.iterations.append(iteration)

    def get_plot_data(self) -> Dict[str, np.ndarray]:
        """Get data ready for plotting."""
        if len(self.train_scores) == 0:
            return {}
            
        return {
            'iterations': np.array(self.iterations),
            'train_scores': np.array(self.train_scores),
            'val_scores': np.array(self.val_scores)
        }

    def __bool__(self) -> bool:
        """Return True if there is data to plot."""
        return len(self.train_scores) > 0 and len(self.val_scores) > 0 and len(self.iterations) > 0

@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc: float = 0.0
    optimal_threshold: float = 0.5
    valid_samples: int = 0
    total_samples: int = 0
    nan_percentage: float = 0.0

def calculate_classification_evaluation_metrics(y_true: np.ndarray, 
                                             y_pred_proba: np.ndarray,
                                             threshold: float = 0.5) -> ClassificationMetrics:
    """
    Calculate classification evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Probability threshold for binary classification
        
    Returns:
        ClassificationMetrics object containing calculated metrics
    """
    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = ClassificationMetrics()
    
    # Calculate basic metrics
    metrics.accuracy = accuracy_score(y_true, y_pred)
    metrics.precision = precision_score(y_true, y_pred)
    metrics.recall = recall_score(y_true, y_pred)
    metrics.f1 = f1_score(y_true, y_pred)
    
    # Calculate AUC
    try:
        metrics.auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        # Handle case where only one class is present
        metrics.auc = 0.0
    
    # Store threshold
    metrics.optimal_threshold = threshold
    
    # Calculate sample statistics
    metrics.valid_samples = np.sum(~np.isnan(y_true))
    metrics.total_samples = len(y_true)
    metrics.nan_percentage = 100 * (1 - metrics.valid_samples / metrics.total_samples)
    
    return metrics
