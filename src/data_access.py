"""
data_access.py

Wrapper functions for saving and loading data

"""

import pandas as pd

from pathlib import Path


DATAPATH = Path(r'data')

# scraped data
NEWLY_SCRAPED_PATH = DATAPATH / 'newly_scraped'
CUMULATIVE_SCRAPED_PATH = DATAPATH / 'cumulative_scraped'
SCRAPED_BOXSCORE_FILES = ["games_traditional.csv", "games_advanced.csv", "games_four-factors.csv", "games_misc.csv", "games_scoring.csv"]
TODAYS_MATCHUPS_FILE = "todays_matchups.csv"
TODAYS_GAMES_IDS_FILE = "todays_games_ids.csv"

def save_data(df: pd.DataFrame, file_name: str, cumulative:bool = False) -> None:
    """
    Saves the dataframe to a csv file in the appropriate directory

    Args:
        df (pd.DataFrame): the scraped data to save
        file_name (str): the name of the file to save the data to
        cumulative (bool): whether to save the newly scraped data or the cumulative scraped data
    """ 

    if cumulative:
        if not CUMULATIVE_SCRAPED_PATH.exists():
            print("Could not find directory for cumulative scraped data")
            exit(1)
        file_path = CUMULATIVE_SCRAPED_PATH 
    else:
        if not NEWLY_SCRAPED_PATH.exists():
            print("Could not find directory for newly scraped data")
            print("Creating directory for newly scraped data")
            NEWLY_SCRAPED_PATH.mkdir(parents=True, exist_ok=True)
        file_path = NEWLY_SCRAPED_PATH

    if file_name == "matchups":
        file_name = TODAYS_MATCHUPS_FILE

    if file_name == "games_ids":
        file_name = TODAYS_GAMES_IDS_FILE
        
    df.to_csv(file_path / file_name, index=False)
    print(f"Data saved to {file_path / file_name}")


def load_scraped_data(cumulative:bool = False) -> list:
    """
    Get the scraped data from the csv files, either the newly scraped data or the cumulative scraped data
    Retrieves all the scraped data from the csv files and returns them as a list of DataFrames

    Args:
        cumulative (bool): whether to load the newly scraped data or the cumulative scraped data

    Returns:
        list: list of DataFrames
    """ 

    if cumulative:
        scraped_path = CUMULATIVE_SCRAPED_PATH
    else:
        scraped_path = NEWLY_SCRAPED_PATH

    all_dfs = []

    if not scraped_path.exists():
        raise FileNotFoundError(f"Directory {scraped_path} not found")
   
    for file in SCRAPED_BOXSCORE_FILES:
        if not (scraped_path / file).exists():
            raise FileNotFoundError(f"File {file} not found in {scraped_path}")
        
        df = pd.read_csv(scraped_path / file)
        
        all_dfs = all_dfs + [df]

    return all_dfs