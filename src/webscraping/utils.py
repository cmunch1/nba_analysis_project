from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone

from .config import config
from ..data_access.data_access import load_scraped_data



def activate_web_driver(browser: str) -> webdriver:
    """
    Activate selenium web driver for use in scraping

    Args:
        browser (str): the name of the browser to use though currently only Chrome is supported

    Returns:
        the selected webdriver
    """
    
    options = [
        "--headless=new",
        "--remote-debugging-port=9222",
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
    ]
    
    chrome_options = webdriver.ChromeOptions() 
    
    for option in options:
        chrome_options.add_argument(option)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)    

    return driver

def get_start_date_and_seasons() -> tuple[datetime, list]:

  
    full_scrape = config["full_scrape"]
    start_season = config["start_season"]
    regular_season_start_month = config["regular_season_start_month"]

    if full_scrape:
        seasons = list(range(start_season, datetime.now().year))
        seasons = [str(season) + "-" + (str(season + 1))[-2:] for season in seasons]
        first_start_date = f"{regular_season_start_month}/1/{start_season}"
    else:
        scraped_data = load_scraped_data(cumulative=True)
        first_start_date, seasons = determine_scrape_start(scraped_data)

        if first_start_date is None:
            print("Error - previous scraped data has inconsistent dates")
            exit()

    return first_start_date, seasons

def determine_scrape_start(scraped_data: list) -> tuple[datetime, list]:
    """
    Determine where to begin scraping for more games based on the latest game in the dataset

    Args:       
        scraped_data (list): list of DataFrames that have been scraped

    Returns:
        tuple: start_date (datetime), seasons (list of ints)
    """ 

    game_date_header_variations = config["game_date_header_variations"] # each column header is spelled differently in different datasets

    # find the last date in the dataset and make sure all the datasets have the same last date
    for i, df in enumerate(scraped_data):
        for date_col in game_date_header_variations:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
                game_date = date_col
                break
        else:
            print(f"Dataframe {i} does not have a recognized game date column")
            return None, None
       
        if i == 0:
            last_date = df[game_date].max()
        else:
            if df[game_date].max() != last_date:
                print(f"Dataframe {i} has a different last date than the first dataframe")
                return None, None

    # determine the season for that date
    if last_date.month >= 10:
        last_season = last_date.year
    else:
        last_season = last_date.year - 1

    # Determine the date of the next day to begin scraping from
    start_date = last_date + timedelta(days=1)
    start_date = start_date.strftime("%m/%d/%Y")

    # determine what season we are in currently
    today = datetime.now(timezone('EST'))
    if today.month >= 10:
        current_season = today.year
    else:
        current_season = today.year - 1

    # determine which seasons we need to scrape to catch up the data
    seasons = list(range(last_season, current_season+1))
    seasons = [str(season) + "-" + (str(season + 1))[-2:] for season in seasons]

    print("Last date in dataset: ", last_date)
    print("Last season in dataset: ", last_season)
    print("Current season: ", current_season)
    print("Seasons to scrape: ", seasons)
    print("Start date: ", start_date)

    return start_date, seasons

def validate_data() -> str:
    """
    Validate the boxscores dataframe
    """
    scraped_data = load_scraped_data(cumulative=False)
    response = validate_scraped_dataframes(scraped_data)

    if response == "Pass":
        print("All scraped dataframes are consistent")
    else:
        print("Error - scraped dataframes are inconsistent")
        print(response)

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


