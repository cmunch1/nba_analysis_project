"""
Configuration Manager for NBA Analysis Project

This module provides a centralized configuration management system for the NBA analysis project.
It reads all YAML configuration files in the same directory and combines them into a single
Config object. This allows for modular and flexible configuration across different aspects
of the project (e.g., web scraping, data processing, analysis parameters).

The Config class automatically loads and parses all YAML files, handles any necessary
data transformations (like unicode character decoding), and exposes all configuration
parameters as attributes for easy access throughout the project.

Usage:
    from config.config import config
    
    # Access configuration parameters
    base_url = config.base_url
    headers = config.headers
"""

import yaml
from pathlib import Path
from types import SimpleNamespace
import unicodedata
import logging
import os
from yaml.constructor import Constructor, ConstructorError

logger = logging.getLogger(__name__)

from .abstract_config import AbstractConfig

class Config(AbstractConfig):
    def __init__(self):
        super().__init__()
        config_dir = Path('.') / 'configs'
        
        # Create a custom SafeLoader class with our constructors
        class CustomSafeLoader(yaml.SafeLoader):
            pass
        
        # Add custom constructors for Optuna suggestion tags
        CustomSafeLoader.add_constructor('!suggest_int', self._construct_suggestion)
        CustomSafeLoader.add_constructor('!suggest_float', self._construct_suggestion)
        
        config_dict = {}
        for config_file in config_dir.glob('*.yaml'):
            with open(config_file, 'r', encoding='utf-8') as yaml_file:
                config_dict.update(yaml.load(yaml_file, Loader=CustomSafeLoader))

        # Convert nested dictionaries to SimpleNamespace recursively
        def dict_to_namespace(d):
            if isinstance(d, str):
                return self.resolve_project_root_path(d)
            if not isinstance(d, dict):
                return d
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})

        config_obj = dict_to_namespace(config_dict)

        # Set all config parameters as attributes of this instance
        for key, value in vars(config_obj).items():
            setattr(self, key, value)
        
        app_config_path = config_dir / 'app_config.yaml'
        if app_config_path.exists():
            with open(app_config_path) as yaml_file:
                app_config = yaml.safe_load(yaml_file)
                for key, value in app_config.items():
                    if isinstance(value, str):
                        value = self.resolve_project_root_path(value)
                    setattr(self, key, value)
        else:
            raise FileNotFoundError(f"app_config.yaml not found in {config_dir}")

    def resolve_project_root_path(self, path: str) -> str:
        """Resolves paths that contain ${PROJECT_ROOT} to absolute paths"""
        if "${PROJECT_ROOT}" in path:
            # Get project root (assuming configs dir is in project root)
            project_root = Path(__file__).parent.parent.parent
            return path.replace("${PROJECT_ROOT}", str(project_root))
        return path

    @staticmethod
    def _construct_suggestion(loader: yaml.Loader, node: yaml.Node) -> dict:
        """Custom constructor for Optuna suggestion tags (!suggest_int and !suggest_float)"""
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        raise ConstructorError(None, None,
            f"expected a mapping node but found {node.id}",
            node.start_mark)





