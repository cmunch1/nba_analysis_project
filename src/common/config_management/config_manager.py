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
from common.app_file_handling.base_app_file_handler import BaseAppFileHandler

logger = logging.getLogger(__name__)

from .base_config_manager import BaseConfigManager

DEFAULT_CONFIG_DIR = Path('..') / 'configs'

class ConfigManager(BaseConfigManager):
    def __init__(self, config_dir: Path = None, app_file_handler: BaseAppFileHandler = None):
        self.config_dir = config_dir or DEFAULT_CONFIG_DIR
        self.app_file_handler = app_file_handler 
        self._load_configurations()

    def get_config(self) -> SimpleNamespace:
        return self

    def _load_configurations(self):
        # Load all YAML files in directory, requiring app_config.yaml
        config_dict = self.app_file_handler.load_yaml_files_in_directory(
            self.config_dir,
            required_files=['app_config.yaml']
        )

        # Convert nested dictionaries to SimpleNamespace recursively
        def dict_to_namespace(d):
            if isinstance(d, str):
                return self.app_file_handler.resolve_project_root_path(d)
            if not isinstance(d, dict):
                return d
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})

        config_obj = dict_to_namespace(config_dict)

        # Set all config parameters as attributes of this instance
        for key, value in vars(config_obj).items():
            setattr(self, key, value)

    @staticmethod
    def _construct_suggestion(loader: yaml.Loader, node: yaml.Node) -> dict:
        """Custom constructor for Optuna suggestion tags (!suggest_int and !suggest_float)"""
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        raise ConstructorError(None, None,
            f"expected a mapping node but found {node.id}",
            node.start_mark)





