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

    # one of the game date headers has a unicode character in it "Game\xa0Date", which causes problems
    config['game_date_header_variations'] = [header.encode().decode('unicode_escape') for header in config['game_date_header_variations']]

    
    return config

config = load_config()