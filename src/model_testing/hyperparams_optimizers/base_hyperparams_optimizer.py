from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from ...common.config_management.base_config_manager import BaseConfigManager
from ...common.app_logging.base_app_logger import BaseAppLogger
from ...common.error_handling.base_error_handler import BaseErrorHandler
from ..hyperparams_managers.base_hyperparams_manager import BaseHyperparamsManager


class BaseHyperparamsOptimizer(ABC):
    @abstractmethod
    def __init__(self,
                 config: BaseConfigManager,
                 optimizer_config: Any,  # Add optimizer-specific config
                 hyperparameter_manager: BaseHyperparamsManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        pass
      
    @property
    @abstractmethod
    def log_performance(self):
        """Get the performance logging decorator from app_logger."""
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
        """
        Return the best parameters found during optimization.
        
        Returns:
            Dictionary of best parameters
        """
        pass