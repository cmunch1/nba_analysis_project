"""
Configuration Manager for NBA Analysis Project

This module provides a centralized configuration management system that supports nested
configuration files across multiple directories. It can handle model-specific configs
in subdirectories while maintaining the existing functionality.
"""

import yaml
from pathlib import Path
from types import SimpleNamespace
import logging
from typing import Dict, Any, Optional
from yaml.constructor import Constructor, ConstructorError
from src.common.app_file_handling.base_app_file_handler import BaseAppFileHandler

logger = logging.getLogger(__name__)

from .base_config_manager import BaseConfigManager

class ConfigManager(BaseConfigManager):
    def __init__(self, config_dir: Path = None, app_file_handler: BaseAppFileHandler = None):
        self.app_file_handler = app_file_handler
        self.config_dir = self._get_default_config_dir() if config_dir is None else config_dir
        self._load_configurations()

    def _get_default_config_dir(self) -> Path:
        """
        Get the default configuration directory from the config_path.yaml file.
        Falls back to a default path if the file cannot be loaded.
        """
        config_path_file = Path(__file__).parent / 'config_path.yaml'
        try:
            print(f"Loading config_path.yaml from: {config_path_file}")
            config_path_data = self.app_file_handler.read_yaml(config_path_file)
            config_path = config_path_data.get('default_config_dir')
            print(f"Config path: {config_path}")
            return Path(config_path)
  
        except Exception as e:
            logger.warning(f"Could not load config_path.yaml: {str(e)}. Using default path.")
            # Fallback to the original hardcoded path
            return Path(__file__).parent.parent.parent / 'configs'

    def get_config(self) -> SimpleNamespace:
        return self

    def _load_configurations(self):
        """
        Load all configuration files, including those in nested directories.
        Handles the new structure with model-specific configs in subdirectories.
        """
        # First load main config files
        print(f"Loading configuration files from: {self.config_dir}")
        config_dict = self.app_file_handler.load_yaml_files_in_directory(
            self.config_dir,
            required_files=['app_config.yaml']
        )

        # Load nested configurations
        nested_configs = self._load_nested_configurations()
        
        # Merge nested configurations into main config
        config_dict = self._merge_configurations(config_dict, nested_configs)

        # Convert to SimpleNamespace and set attributes
        config_obj = self._dict_to_namespace(config_dict)
        for key, value in vars(config_obj).items():
            setattr(self, key, value)

    def _load_nested_configurations(self) -> Dict[str, Any]:
        """
        Recursively load configurations from nested directories.
        Returns a dictionary with nested configuration data.
        """
        nested_configs = {}
        
        for item in self.config_dir.rglob('*.yaml'):
            # Skip files in the root config directory
            if item.parent == self.config_dir:
                continue
                
            try:
                # Create nested dictionary structure based on directory path
                current_level = nested_configs
                relative_path = item.relative_to(self.config_dir)
                parts = list(relative_path.parts[:-1])  # Exclude filename
                
                # Build nested structure
                for part in parts:
                    current_level = current_level.setdefault(part, {})
                
                # Load and add config file content
                config_content = self.app_file_handler.read_yaml(item)
                key = item.stem
                current_level[key] = config_content
                
                logger.debug(f"Loaded nested configuration from: {item}")
            except Exception as e:
                logger.error(f"Error loading nested configuration {item}: {str(e)}")
                raise

        return nested_configs

    def _merge_configurations(self, main_config: Dict[str, Any], 
                            nested_configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge nested configurations into the main configuration dictionary.
        Handles potential naming conflicts and maintains structure.
        """
        merged = main_config.copy()
        
        # Create models section if it doesn't exist
        if 'models' not in merged and 'models' in nested_configs:
            merged['models'] = {}
            
        # Merge nested configurations
        for key, value in nested_configs.items():
            if isinstance(value, dict):
                if key not in merged:
                    merged[key] = {}
                merged[key] = self._deep_merge(merged[key], value)
                
        return merged

    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two dictionaries, preserving nested structures.
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result

    def _dict_to_namespace(self, d: Dict[str, Any]) -> Any:
        """
        Recursively convert a dictionary to SimpleNamespace.
        Handles path resolution for string values.
        """
        if isinstance(d, str):
            return self.app_file_handler.resolve_project_root_path(d)
        if not isinstance(d, dict):
            return d
        return SimpleNamespace(**{k: self._dict_to_namespace(v) for k, v in d.items()})

    @staticmethod
    def _construct_suggestion(loader: yaml.Loader, node: yaml.Node) -> dict:
        """Custom constructor for Optuna suggestion tags."""
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        raise ConstructorError(None, None,
            f"expected a mapping node but found {node.id}",
            node.start_mark)





