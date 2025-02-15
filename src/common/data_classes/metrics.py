"""Data classes for model metrics and learning curves."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np

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
        if not self.train_scores:
            return {}
            
        return {
            'iterations': np.array(self.iterations),
            'train_scores': np.array(self.train_scores),
            'val_scores': np.array(self.val_scores)
        }

    def __bool__(self) -> bool:
        """Return True if there is data to plot."""
        return bool(self.train_scores and self.val_scores and self.iterations)

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
