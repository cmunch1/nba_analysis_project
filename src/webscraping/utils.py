from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone

GAME_DATE_VARIATIONS = ["Game Date", "Game_Date", "GAME DATE", "Game\xa0Date"]

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

def determine_scrape_start(scraped_data: list) -> tuple[datetime, list]:
    """
    Determine where to begin scraping for more games based on the latest game in the dataset

    Args:       
        scraped_data (list): list of DataFrames that have been scraped

    Returns:
        tuple: start_date (datetime), seasons (list of ints)
    """ 

    # find the last date in the dataset and make sure all the datasets have the same last date
    for i, df in enumerate(scraped_data):
        for date_col in GAME_DATE_VARIATIONS:
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