from enum import Enum
from typing import Type, Dict

from src.common.config_management.base_config_manager import BaseConfigManager
from src.common.app_logging.base_app_logger import BaseAppLogger
from src.common.error_handling.base_error_handler import BaseErrorHandler
from src.common.app_file_handling.base_app_file_handler import BaseAppFileHandler

from .base_hyperparams_optimizer import BaseHyperparamsOptimizer
from .optuna_optimizer import OptunaOptimizer
from ..hyperparams_managers.base_hyperparams_manager import BaseHyperparamsManager


class OptimizerType(Enum):
    """Supported hyperparameter optimizer types."""
    OPTUNA = "optuna"
    # Add more optimizer types as needed
    # HYPEROPT = "hyperopt"
    # RAY_TUNE = "ray_tune"
    # BOHB = "bohb"

class OptimizerFactory:
    """Factory for creating hyperparameter optimizers with proper dependency injection."""
    
    # Registry of available optimizers
    _optimizers: Dict[OptimizerType, Type[BaseHyperparamsOptimizer]] = {
        OptimizerType.OPTUNA: OptunaOptimizer,
        # Add more mappings as you create new optimizers
        # OptimizerType.HYPEROPT: HyperoptOptimizer,
        # OptimizerType.RAY_TUNE: RayTuneOptimizer,
        # OptimizerType.BOHB: BOHBOptimizer,
    }

    @classmethod
    def create_optimizer(cls,
                        optimizer_type: OptimizerType,
                        config: BaseConfigManager,
                        hyperparameter_manager: BaseHyperparamsManager,
                        app_logger: BaseAppLogger,
                        error_handler: BaseErrorHandler,
                        app_file_handler: BaseAppFileHandler) -> BaseHyperparamsOptimizer:
        """Create optimizer with proper nested config handling."""
        optimizer_class = cls._optimizers.get(optimizer_type)
        if optimizer_class is None:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
        
        return optimizer_class(
            config=config,
            hyperparameter_manager=hyperparameter_manager,
            app_logger=app_logger,
            error_handler=error_handler
        )


    @classmethod
    def register_optimizer(cls,
                         optimizer_type: OptimizerType,
                         optimizer_class: Type[BaseHyperparamsOptimizer]) -> None:
        """
        Register a new optimizer type.
        
        Args:
            optimizer_type: Enum value for the optimizer type
            optimizer_class: Optimizer class to register
        """
        if not issubclass(optimizer_class, BaseHyperparamsOptimizer):
            raise TypeError(
                f"Optimizer class must inherit from BaseHyperparamsOptimizer, "
                f"got {optimizer_class.__name__}"
            )
        cls._optimizers[optimizer_type] = optimizer_class

    @classmethod
    def list_available_optimizers(cls) -> list[str]:
        """
        Get a list of available optimizer types.
        
        Returns:
            List of optimizer type names
        """
        return [opt.value for opt in cls._optimizers.keys()]