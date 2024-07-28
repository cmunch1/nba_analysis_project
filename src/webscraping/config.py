import json
from pathlib import Path


# config file location
webscraping_dir = Path(__file__).parent
config_path = webscraping_dir / "config.json"

def load_config() -> dict:
    """
    Load the webscraping configuration file

    """
    if Path(config_path).exists() == False:
        raise Exception(f"{config_path} file not found")

    with open(config_path) as json_file:
        config = json.load(json_file)
    
    return config

config = load_config()