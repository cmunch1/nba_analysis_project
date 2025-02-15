from abc import ABC, abstractmethod
from ...common.config_management.base_config_manager import BaseConfigManager    
import pandas as pd

class BaseExperimentLogger(ABC):
    @abstractmethod
    def __init__(self, config: BaseConfigManager):
        pass

    @abstractmethod
    def log_experiment(self, experiment_name: str, experiment_description: str, model_name: str, 
                      model: object, model_params: dict, oof_metrics: dict, validation_metrics: dict, 
                      oof_data: pd.DataFrame, val_data: pd.DataFrame):
        pass

    @abstractmethod
    def log_model(self, model: object, model_name: str, model_params: dict):
        pass