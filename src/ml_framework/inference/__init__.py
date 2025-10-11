"""
Inference module for loading and using trained models.

Provides a clean interface for making predictions using models stored
in a model registry (MLflow, custom, etc.) with automatic preprocessing.
"""

from .model_predictor import ModelPredictor

__all__ = ['ModelPredictor']
