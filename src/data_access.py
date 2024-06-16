import pandas as pd

SCRAPED_FILES = ["games_traditional.csv", "games_advanced.csv", "games_four-factors.csv", "games_misc.csv", "games_scoring.csv"]

from pathlib import Path  #for Windows/Linux compatibility
DATAPATH = Path(r'data')


def load_scraped_data() -> list:
    """
    Get the scraped data from the csv files
    """     

    all_dfs = []
   
    for file in SCRAPED_FILES:
        if not (DATAPATH / file).exists():
            raise FileNotFoundError(f"File {file} not found in {DATAPATH}")
        
        df = pd.read_csv(DATAPATH / file)
        
        all_dfs = all_dfs + [df]

    return all_dfs