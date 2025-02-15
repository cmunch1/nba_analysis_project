from abc import ABC, abstractmethod
from ...common.config_management.base_config_manager import BaseConfigManager
from typing import Dict, Any, Optional, Callable

class BaseHyperparameterOptimizer(ABC):
    """Abstract base class for hyperparameter optimizers."""
    
    @abstractmethod
    def __init__(self, config: BaseConfigManager):
        pass
      
    @abstractmethod
    def optimize(self, 
                objective_func: Optional[Callable] = None,
                param_space: Dict[str, Any] = None,
                X = None,
                y = None,
                model_type: str = None,
                cv: int = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters using the specified objective function.
        
        Args:
            objective_func: Optional custom objective function
            param_space: Dictionary defining parameter search space
            X: Training features
            y: Training labels
            model_type: Type of model to optimize
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary containing best parameters and optimization results
        """
        pass
    
    @abstractmethod
    def get_best_params(self) -> Dict[str, Any]:
        """Return the best parameters found during optimization."""
        pass