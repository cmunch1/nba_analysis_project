"""Data classes for preprocessing tracking."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd
import json

@dataclass
class PreprocessingStep:
    """Records information about a single preprocessing step"""
    name: str
    type: str
    columns: List[str]
    parameters: Dict[str, Any]
    statistics: Dict[str, Any] = field(default_factory=dict)
    output_features: List[str] = field(default_factory=list)

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
            'dropped_features': list(self.dropped_features),
            'engineered_features': self.engineered_features,
            'feature_transformations': self.feature_transformations,
            'final_shape': self.final_shape
        }

    def to_json(self) -> str:
        """Convert preprocessing results to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
