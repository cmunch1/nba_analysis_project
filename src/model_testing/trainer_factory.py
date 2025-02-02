from enum import Enum
from typing import Dict, Type
from .trainers import (
    BaseTrainer,
    XGBoostTrainer,
    LightGBMTrainer,
    SKLearnTrainer,
    CatBoostTrainer
)


class TrainerType(Enum):
    XGBOOST = "XGBoost"
    LIGHTGBM = "LGBM"
    SKLEARN = "SKLearn"
    CATBOOST = "CatBoost"

class TrainerFactory:
    _trainer_map: Dict[TrainerType, Type[BaseTrainer]] = {
        TrainerType.XGBOOST: XGBoostTrainer,
        TrainerType.LIGHTGBM: LightGBMTrainer,
        TrainerType.SKLEARN: SKLearnTrainer,
        TrainerType.CATBOOST: CatBoostTrainer,
    }

    @classmethod
    def create_trainers(cls, config) -> Dict[str, BaseTrainer]:
        """Creates trainers based on models specified in configuration."""
        trainers = {}
        
        # Process each model in the config
        for model_name in vars(config.models):
            enabled = getattr(config.models, model_name)
            
            # Handle nested SKLearn models
            if model_name == "SKLearn":
                if hasattr(enabled, '__dict__'):  # Check if it's a namespace
                    for sklearn_model, is_enabled in vars(enabled).items():
                        if is_enabled:
                            trainer_type = TrainerType.SKLEARN
                            trainer_class = cls._trainer_map[trainer_type]
                            trainers[f"SKLearn_{sklearn_model}"] = trainer_class(config, sklearn_model)
            # Handle other model types
            elif enabled:
                try:
                    trainer_type = TrainerType[model_name.upper()]
                    trainer_class = cls._trainer_map[trainer_type]
                    trainers[model_name] = trainer_class(config)
                except KeyError:
                    raise ValueError(f"Unsupported model type: {model_name}")
        
        return trainers 