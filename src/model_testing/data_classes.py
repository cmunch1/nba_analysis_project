"""
Data classes for model testing

Contains:
- ClassificationMetrics: Metrics for classification models
- PreprocessingStep: Information about a single preprocessing step
- PreprocessingInfo: Tracks all preprocessing steps and their effects
- ModelTrainingResults: Comprehensive container for model training results, evaluation metrics,
  and preprocessing tracking.

"""



from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Set
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import json
@dataclass
class ClassificationMetrics:
    def __init__(self):
        self.accuracy: float = 0.0
        self.precision: float = 0.0
        self.recall: float = 0.0
        self.f1: float = 0.0
        self.auc: float = 0.0
        self.optimal_threshold: float = 0.0


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
            'parameters': self.parameters,
            'statistics': {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.statistics.items()
            },
            'output_features': self.output_features
        }

@dataclass
class PreprocessingInfo:
    """Tracks all preprocessing steps and their effects"""
    steps: List[PreprocessingStep] = field(default_factory=list)
    original_features: List[str] = field(default_factory=list)
    final_features: List[str] = field(default_factory=list)
    dropped_features: Set[str] = field(default_factory=set)
    engineered_features: Dict[str, List[str]] = field(default_factory=dict)  # Maps new features to their source features
    
    def add_step(self, step: PreprocessingStep) -> None:
        """Add a preprocessing step and update feature tracking"""
        self.steps.append(step)
        # Update final features based on the step's output
        if step.output_features:
            self.final_features = step.output_features
        # Track dropped features
        current_features = set(step.output_features)
        previous_features = set(self.final_features or self.original_features)
        self.dropped_features.update(previous_features - current_features)

    def get_feature_lineage(self, feature_name: str) -> List[str]:
        """Get the preprocessing history of a specific feature"""
        lineage = []
        for step in self.steps:
            if feature_name in step.output_features:
                lineage.append(f"{step.type}:{step.name}")
        return lineage

    def to_dict(self) -> Dict[str, Any]:
        """Convert preprocessing info to a dictionary for serialization"""
        return {
            'steps': [step.to_dict() for step in self.steps],
            'original_features': self.original_features,
            'final_features': self.final_features,
            'dropped_features': list(self.dropped_features),
            'engineered_features': self.engineered_features
        }

    def to_json(self) -> str:
        """Convert preprocessing info to JSON string"""
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
        self.metrics: Optional[ClassificationMetrics] = None
        
        # Data fields
        self.feature_data: Optional[pd.DataFrame] = None
        self.target_data: Optional[pd.Series] = None
        self.binary_predictions: Optional[NDArray[np.int_]] = None
        self.probability_predictions: Optional[NDArray[np.float_]] = None
        
        # Evaluation context
        self.is_validation: bool = False
        self.evaluation_type: str = ""

        # Reference to preprocessing results
        self.preprocessing_results: Optional[PreprocessingResults] = None

    def update_feature_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Update the feature and target data stored in the results.
        
        Args:
            X: Feature DataFrame
            y: Target Series
        """
        self.feature_data = X
        self.target_data = y

    def update_predictions(self, probability_predictions: NDArray[np.float_], threshold: float = 0.5) -> None:
        """
        Update predictions with both probability scores and binary predictions.
        
        Args:
            probability_predictions: Array of probability scores
            threshold: Classification threshold (default: 0.5)
        """
        self.probability_predictions = probability_predictions
        self.binary_predictions = (probability_predictions >= threshold).astype(int)
        self.predictions = probability_predictions  # For backward compatibility

    def add_preprocessing_step(self, 
                             name: str,
                             step_type: str,
                             columns: List[str],
                             parameters: Dict[str, Any],
                             statistics: Dict[str, Any] = None,
                             output_features: List[str] = None) -> None:
        """
        Add a preprocessing step to the tracking.
        
        Args:
            name: Name of the preprocessing step
            step_type: Type of preprocessing
            columns: Columns the step was applied to
            parameters: Parameters used for the step
            statistics: Optional learned statistics
            output_features: Optional list of output feature names
        """
        step = PreprocessingStep(
            name=name,
            type=step_type,
            columns=columns,
            parameters=parameters,
            statistics=statistics or {},
            output_features=output_features or []
        )
        self.preprocessing_info.add_step(step)

    def get_feature_preprocessing_history(self, feature_name: str) -> List[Dict[str, Any]]:
        """
        Get the preprocessing history for a specific feature.
        
        Args:
            feature_name: Name of the feature to get history for
            
        Returns:
            List of preprocessing steps that affected this feature
        """
        history = []
        for step in self.preprocessing_info.steps:
            if feature_name in step.columns or feature_name in step.output_features:
                history.append(step.to_dict())
        return history

    def prepare_for_logging(self) -> Dict[str, Any]:
        """
        Prepare the results for experiment logging by organizing data into a structured format.
        Now includes preprocessing information.
        """
        prefix = "val_" if self.is_validation else "oof_"
        
        # Prepare the full dataframe with predictions
        if self.feature_data is not None and self.target_data is not None:
            full_data = self.feature_data.copy()
            full_data['target'] = self.target_data
            full_data[f'{prefix}predictions'] = self.probability_predictions
        else:
            full_data = None

        return {
            f"{prefix}data": full_data,
            f"{prefix}metrics": self.metrics,
            "model_name": self.model_name,
            "model": self.model,
            "model_params": self.model_params,
            "preprocessing_info": self.preprocessing_info.to_dict(),
            "preprocessing_config": self.preprocessing_config
        }

    def get_feature_transformations(self) -> Dict[str, List[str]]:
        """
        Get a mapping of original features to their transformed versions.
        
        Returns:
            Dict mapping original feature names to list of transformed feature names
        """
        transformations = {}
        for original in self.preprocessing_info.original_features:
            transformed = []
            for step in self.preprocessing_info.steps:
                if original in step.columns:
                    transformed.extend([f for f in step.output_features 
                                     if f.startswith(original)])
            if transformed:
                transformations[original] = transformed
        return transformations

    def summarize_preprocessing(self) -> Dict[str, Any]:
        """
        Generate a summary of all preprocessing steps and their effects.
        
        Returns:
            Dict containing preprocessing summary statistics
        """
        return {
            'n_original_features': len(self.preprocessing_info.original_features),
            'n_final_features': len(self.preprocessing_info.final_features),
            'n_dropped_features': len(self.preprocessing_info.dropped_features),
            'n_preprocessing_steps': len(self.preprocessing_info.steps),
            'preprocessing_steps': [
                f"{step.type}:{step.name}" for step in self.preprocessing_info.steps
            ],
            'dropped_features': list(self.preprocessing_info.dropped_features),
            'engineered_features': self.preprocessing_info.engineered_features
        }