"""
Data classes for model testing and preprocessing tracking.

Contains:
- ClassificationMetrics: Metrics for classification models
- PreprocessingStep: Information about a single preprocessing step
- PreprocessingResults: Complete preprocessing tracking information
- ModelTrainingResults: Container for model training results and evaluation metrics
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Set, Tuple, Union
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import json

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

@dataclass
class PreprocessingStep:
    """Records information about a single preprocessing step"""
    name: str  # Name of the preprocessing step (e.g., 'StandardScaler', 'OneHotEncoder')
    type: str  # Type of preprocessing ('numerical', 'categorical', 'feature_selection', etc.)
    columns: List[str]  # Columns this step was applied to
    parameters: Dict[str, Any]  # Parameters used for this step
    statistics: Dict[str, Any] = field(default_factory=dict)  # Learned statistics (e.g., mean, std)
    output_features: List[str] = field(default_factory=list)  # New feature names after transformation

    def to_dict(self) -> Dict[str, Any]:
        """Convert the preprocessing step to a dictionary for serialization"""
        return {
            'name': self.name,
            'type': self.type,
            'columns': self.columns,
            'parameters': self._convert_to_serializable(self.parameters),
            'statistics': self._convert_to_serializable(self.statistics),
            'output_features': self.output_features
        }

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects to serializable formats"""
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return obj


@dataclass
class PreprocessingResults:
    """Tracks all preprocessing information and transformations"""
    steps: List[PreprocessingStep] = field(default_factory=list)
    original_features: List[str] = field(default_factory=list)
    final_features: List[str] = field(default_factory=list)
    dropped_features: Set[str] = field(default_factory=set)
    engineered_features: Dict[str, List[str]] = field(default_factory=dict)
    feature_transformations: Dict[str, List[str]] = field(default_factory=dict)
    final_shape: Optional[Tuple[int, int]] = None

    def add_step(self, step: PreprocessingStep) -> None:
        """Add a preprocessing step and update feature tracking"""
        self.steps.append(step)
        
        # Update final features based on the step's output
        if step.output_features:
            self.final_features = step.output_features
            
        # Track dropped features
        current_features = set(step.output_features) if step.output_features else set()
        previous_features = set(self.final_features or self.original_features)
        self.dropped_features.update(previous_features - current_features)

    def get_feature_lineage(self, feature_name: str) -> List[str]:
        """Get the preprocessing history of a specific feature"""
        lineage = []
        for step in self.steps:
            if feature_name in step.columns or feature_name in step.output_features:
                lineage.append(f"{step.type}:{step.name}")
        return lineage

    def summarize(self) -> Dict[str, Any]:
        """Generate a summary of preprocessing results"""
        return {
            'n_original_features': len(self.original_features),
            'n_final_features': len(self.final_features),
            'n_dropped_features': len(self.dropped_features),
            'n_preprocessing_steps': len(self.steps),
            'preprocessing_steps': [f"{step.type}:{step.name}" for step in self.steps],
            'dropped_features': list(self.dropped_features),
            'engineered_features': self.engineered_features,
            'final_shape': self.final_shape
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert preprocessing results to a dictionary for serialization"""
        return {
            'steps': [step.to_dict() for step in self.steps],
            'original_features': self.original_features,
            'final_features': self.final_features,
            'dropped_features': list(self.dropped_features),  # Convert set to list for JSON serialization
            'engineered_features': self.engineered_features,
            'feature_transformations': self.feature_transformations,
            'final_shape': self.final_shape
        }

    def to_json(self) -> str:
        """Convert preprocessing results to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ModelTrainingResults:
    """Comprehensive container for model training results and evaluation metrics"""
    def __init__(self, X_shape: tuple[int, int]):
        # Model-specific fields
        self.predictions: Optional[NDArray[np.float_]] = None
        self.shap_values: Optional[NDArray[np.float_]] = None
        self.shap_interaction_values: Optional[NDArray[np.float_]] = None
        self.feature_names: List[str] = []
        self.feature_importance_scores: Optional[NDArray[np.float_]] = None
        self.model: Optional[Any] = None
        self.model_name: str = ""
        self.model_params: Dict = {}
        self.num_boost_round: int = 0
        self.early_stopping: int = 0
        self.enable_categorical: bool = False
        self.categorical_features: List[str] = []
        self.metrics: Optional[ClassificationMetrics] = None
        self.eval_metric = None  
        
        # Data fields - Initialize with empty DataFrame of correct shape
        self.feature_data: Optional[pd.DataFrame] = pd.DataFrame(np.zeros(X_shape))  # Initialize with zeros
        self.target_data: Optional[pd.Series] = None
        self.binary_predictions: Optional[NDArray[np.int_]] = None
        self.probability_predictions: Optional[NDArray[np.float_]] = None
        
        # Evaluation context
        self.is_validation: bool = False
        self.evaluation_type: str = ""
        
        # Preprocessing results
        self.preprocessing_results: Optional[PreprocessingResults] = None
        
        self.learning_curve_data = LearningCurveData()
        self.n_folds = 0

    def add_learning_curve_point(self, train_size: int, train_score: float, val_score: float, 
                               fold: int, iteration: Optional[int] = None, metric_name: Optional[str] = None):
        """Add a learning curve data point."""
        self.learning_curve_data.add_point(
            train_size=train_size,
            train_score=train_score,
            val_score=val_score,
            fold=fold,
            iteration=iteration,
            metric_name=metric_name
        )

    def update_feature_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Update feature and target data"""
        self.feature_data = X
        self.target_data = y
        self.feature_names = X.columns.tolist()

    def update_predictions(self, probability_predictions: NDArray[np.float_], 
                         threshold: float = 0.5) -> None:
        """Update probability and binary predictions"""
        self.probability_predictions = probability_predictions
        self.binary_predictions = (probability_predictions >= threshold).astype(int)
        self.predictions = probability_predictions  # For backward compatibility

    def prepare_for_logging(self) -> Dict[str, Any]:
        """Prepare results for experiment logging"""
        prefix = "val_" if self.is_validation else "oof_"
        
        # Prepare full dataframe with predictions
        if self.feature_data is not None and self.target_data is not None:
            full_data = self.feature_data.copy()
            full_data['target'] = self.target_data
            if self.probability_predictions is not None:
                full_data[f'{prefix}predictions'] = self.probability_predictions
        else:
            full_data = None

        return {
            f"{prefix}data": full_data,
            f"{prefix}metrics": self.metrics,
            "model_name": self.model_name,
            "model": self.model,
            "model_params": self.model_params,
            "preprocessing_results": self.preprocessing_results.to_dict() if self.preprocessing_results else None
        }

    def prepare_for_charting(self) -> Dict[str, Any]:
        """Prepare data for chart generation"""
        return {
            "feature_importance": self.feature_importance_scores,
            "feature_names": self.feature_names,
            "y_true": self.target_data,
            "y_pred": self.binary_predictions,
            "y_prob": self.probability_predictions,
            "model": self.model,
            "X": self.feature_data,
            "prefix": "val_" if self.is_validation else "oof_"
        }