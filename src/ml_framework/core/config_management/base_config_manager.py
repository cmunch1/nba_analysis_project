from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any

class BaseConfigManager(ABC):
    @abstractmethod
    def __init__(self, config_dir: Path = None, app_file_handler = None):
        """
        Initialize the config manager with an optional config directory and file handler
        
        Args:
            config_dir (Path, optional): Directory containing configuration files
            app_file_handler: File handler for loading configuration files
        """
        pass

    @abstractmethod
    def get_config(self) -> SimpleNamespace:
        """Get the complete configuration object"""
        pass

    @abstractmethod
    def _load_configurations(self) -> None:
        """Load all configuration files and set up the configuration object"""
        pass

    @abstractmethod
    def _load_nested_configurations(self) -> Dict[str, Any]:
        """
        Load configurations from nested subdirectories
        
        Returns:
            Dict[str, Any]: Dictionary containing nested configuration data
        """
        pass

    @abstractmethod
    def _merge_configurations(self, main_config: Dict[str, Any], 
                            nested_configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge nested configurations with the main configuration
        
        Args:
            main_config (Dict[str, Any]): Main configuration dictionary
            nested_configs (Dict[str, Any]): Nested configurations dictionary
            
        Returns:
            Dict[str, Any]: Merged configuration dictionary
        """
        pass

    @abstractmethod
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two dictionaries
        
        Args:
            dict1 (Dict[str, Any]): First dictionary
            dict2 (Dict[str, Any]): Second dictionary
            
        Returns:
            Dict[str, Any]: Merged dictionary
        """
        pass

    @abstractmethod
    def _dict_to_namespace(self, d: Dict[str, Any]) -> Any:
        """
        Convert a dictionary to SimpleNamespace recursively
        
        Args:
            d (Dict[str, Any]): Dictionary to convert
            
        Returns:
            Any: Converted SimpleNamespace or original value
        """
        pass
