from abc import ABC, abstractmethod
from ...common.config_management.base_config_manager import BaseConfigManager
from typing import Dict, Any, Optional

class BaseHyperparamsManager(ABC):
    @abstractmethod
    def get_current_params(self, model_name: str) -> Dict[str, Any]:
        pass    
    
    @abstractmethod
    def update_best_params(self, model_name: str, new_params: Dict[str, Any], metrics: Dict[str, float], experiment_id: str, run_id: str, description: Optional[str] = None) -> None:
        pass