from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace

class BaseConfigManager(ABC):
    @abstractmethod
    def __init__(self, config_dir: Path = None):
        """Initialize the config manager with an optional config directory"""
        pass

    @abstractmethod
    def get_config(self) -> SimpleNamespace:
        """Get the complete configuration object"""
        pass

    @abstractmethod
    def resolve_project_root_path(self, path: str) -> str:
        """Resolve paths containing ${PROJECT_ROOT} to absolute paths"""
        pass

