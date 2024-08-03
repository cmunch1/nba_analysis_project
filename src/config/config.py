"""
Configuration Manager for NBA Analysis Project

This module provides a centralized configuration management system for the NBA analysis project.
It reads all JSON configuration files in the same directory and combines them into a single
Config object. This allows for modular and flexible configuration across different aspects
of the project (e.g., web scraping, data processing, analysis parameters).

The Config class automatically loads and parses all JSON files, handles any necessary
data transformations (like unicode character decoding), and exposes all configuration
parameters as attributes for easy access throughout the project.

Usage:
    from config.config import config
    
    # Access configuration parameters
    base_url = config.base_url
    headers = config.headers
"""


import json
from pathlib import Path
from types import SimpleNamespace

class Config:
    def __init__(self):
        config_dir = Path(__file__).parent
        
        config_dict = {}
        for config_file in config_dir.glob('*.json'):
            with open(config_file) as json_file:
                config_dict.update(json.load(json_file))

        # Convert the dictionary to an object with attributes
        config_obj = SimpleNamespace(**config_dict)

        # Handle the unicode character in game date headers
        if hasattr(config_obj, 'game_date_header_variations'):
            config_obj.game_date_header_variations = [
                header.encode().decode('unicode_escape')
                for header in config_obj.game_date_header_variations
            ]

        # Set all config parameters as attributes of this instance
        for key, value in vars(config_obj).items():
            setattr(self, key, value)

# Create a single instance of Config
config = Config()


