import pandas as pd
import numpy as np

def validate_scraped_dataframes(scraped_dataframes: list) -> str:
    response = "Pass"
    num_rows = 0
    game_ids = None
    
    for i, df in enumerate(scraped_dataframes):
        if df.duplicated().any():
            return f"Dataframe {i} has duplicate records"
        
        if df.isnull().values.any():
            return f"Dataframe {i} has null values"

        df = df.sort_values(by='GAME_ID')

        if i == 0:
            num_rows = df.shape[0]
            game_ids = df['GAME_ID']
        else:
            if num_rows != df.shape[0]:
                return f"Dataframe {i} does not match the number of rows of the first dataframe"
            
            if not np.array_equal(game_ids.values, df['GAME_ID'].values):
                return f"Dataframe {i} does not match the game ids of the first dataframe"
            
    return response