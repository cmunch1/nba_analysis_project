from enum import Enum
from typing import Type
from ..config.config import AbstractConfig
from .abstract_model_testing import AbstractHyperparameterOptimizer, AbstractHyperparameterManager
from .optimizers.optuna_optimizer import OptunaOptimizer

class OptimizerType(Enum):
    OPTUNA = "optuna"
    # Add more optimizer types as needed
    # HYPEROPT = "hyperopt"
    # RAY_TUNE = "ray_tune"

class OptimizerFactory:
    _optimizers = {
        OptimizerType.OPTUNA: OptunaOptimizer,
        # Add more mappings as you create new optimizers
        # OptimizerType.HYPEROPT: HyperoptOptimizer,
        # OptimizerType.RAY_TUNE: RayTuneOptimizer,
    }

    @classmethod
    def create_optimizer(cls, optimizer_type: OptimizerType, config: AbstractConfig, hyperparameter_manager: AbstractHyperparameterManager) -> AbstractHyperparameterOptimizer:
        optimizer_class = cls._optimizers.get(optimizer_type)
        if optimizer_class is None:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        return optimizer_class(config, hyperparameter_manager)

    @classmethod
    def register_optimizer(cls, optimizer_type: OptimizerType, optimizer_class: Type[AbstractHyperparameterOptimizer]):
        cls._optimizers[optimizer_type] = optimizer_class