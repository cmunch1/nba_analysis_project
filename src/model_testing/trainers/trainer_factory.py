from enum import Enum
from typing import Dict, Type
from . import (
    BaseTrainer,
    XGBoostTrainer,
    LightGBMTrainer,
    SKLearnTrainer,
    CatBoostTrainer
)
from src.common.config_management.base_config_manager import BaseConfigManager
from src.common.app_logging.base_app_logger import BaseAppLogger
from src.common.error_handling.base_error_handler import BaseErrorHandler


class TrainerType(Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    SKLEARN = "sklearn"
    CATBOOST = "catboost"

class TrainerFactory:
    _trainer_map: Dict[TrainerType, Type[BaseTrainer]] = {
        TrainerType.XGBOOST: XGBoostTrainer,
        TrainerType.LIGHTGBM: LightGBMTrainer,
        TrainerType.SKLEARN: SKLearnTrainer,
        TrainerType.CATBOOST: CatBoostTrainer,
    }

    @classmethod
    def create_trainers(cls, config: BaseConfigManager, app_logger: BaseAppLogger, error_handler: BaseErrorHandler) -> Dict[str, BaseTrainer]:
        """Creates trainers based on models specified in configuration."""
        trainers = {}
        
        # Process each model in the config
        for model_name in vars(config.models):
            enabled = getattr(config.models, model_name)

            # Extract the base model name (strip '_config' suffix if present)
            base_model_name = model_name.split('_')[0].upper()
            
            # Handle sklearn models
            if model_name.lower().startswith('sklearn_'):
                if enabled:
                    trainer_type = TrainerType.SKLEARN
                    trainer_class = cls._trainer_map[trainer_type]
                    # Extract the specific sklearn model type from the name
                    sklearn_model_type = model_name.lower().replace('sklearn_', '')
                    trainers[model_name.upper()] = trainer_class(
                        config, 
                        app_logger, 
                        error_handler,
                        model_type=sklearn_model_type
                    )
            # Handle other model types
            elif enabled:
                try:
                    trainer_type = TrainerType[base_model_name]
                    trainer_class = cls._trainer_map[trainer_type]
                    trainers[base_model_name] = trainer_class(config, app_logger, error_handler)
                except KeyError:
                    raise ValueError(f"Unsupported model type: {base_model_name}")
        
        return trainers 