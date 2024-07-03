"""
webscraping.py

This module scrapes the NBA.com website for data/statistics from previous games, and it 
finds the team ids and game ids for games scheduled today.

Selenium and BeautifulSoup do all the heavy lifting in this module. Selenium is required because the data is dynamic and requires javascript to load.

(If running from GitHub Actions or other public servers, a proxy service may be required to avoid being blocked by nba.com.)

This module saves this freshly scraped data to csv files in a specific data directory for later processing. The main
purpose is to simply get the data from the website and save the fairly raw data to a convenient location for later processing.
Data manipulation and cleaning is minimal in this module, but after the csv files are saved, it does check for nulls and
that the data is consistent across the different csv files (e.g. same games in each file, same number of rows) to make sure the scraping went well.

Because of the way the data is structured on nba.com, the data is scraped in chunks based on:
    type of stats ('traditional', 'advanced', 'four-factors', 'misc', 'scoring,)
    season, 
    sub-season (regular season, play-in, playoffs), 
    and date range.

The search function on nba.com requires each of these be specified, and the data is served on different pages for each of these.

(For example, the traditional stats for the 2021 regular season are on a different page than the traditional stats for the 2021 playoffs when
using the search function on nba.com.)

The data from different seasons, sub-seasons, and date ranges, while retrieved in chunks, all have the same columns for each stat type and 
are combined (concatenated) into a single DataFrame for each type of stats.

The different stat pages (traditional, advanced, etc..) all have different columns, so the data is saved to separate csv files for each type of stats,
and the merging of the data into one single big table will be done later.

The data is saved to multiple csv files (depending on the of type stat page it came from) that will be joined later.
    games_traditional.csv, 
    games_advanced.csv
    games_four-factors.csv, 
    games_misc.csv, 
    games_scoring.csv

We also need to scrape the matchups for games scheduled today, so we can predict the winner before the game is played.
The boxscores pages are not available until after the game is played, so we have to create our own records of the games scheduled for today:
    go to the schedule page, 
    retrieve the team ids for each matchup, 
    retrieve the game ids for each game,
    and save these to csv files:

    todays_matchups.csv
    todays_game_ids.csv
    
This will enable us to later construct records of the games that are scheduled for today into our feature store/database, and then 
classify these games as wins or losses.

    
** These are all basically temporary raw data files that will be processed further in other modules. **

"""

import pandas as pd
import numpy as np

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType

from bs4 import BeautifulSoup as soup

import time
from datetime import datetime, timedelta
from pytz import timezone


from src.constants import (
    OFF_SEASON_START, 
    REGULAR_SEASON_START, 
    PLAYOFFS_START,
)   

from src.data_access import (
    load_scraped_data, 
    save_scraped_data,
)


NBA_SCHEDULE = "https://www.nba.com/schedule"
NBA_BOXSCORES = "https://www.nba.com/stats/teams/boxscores"
SUB_SEASON_TYPES = ["Regular+Season", "PlayIn", "Playoffs"] #nba.com has different boxscore pages for each of these sub-seasons
STAT_TYPES = ['traditional', 'advanced', 'four-factors', 'misc', 'scoring'] #nba.com has different boxscore pages for each of these stat types

GAME_DATE_VARIATIONS = ["Game Date", "Game_Date", "GAME DATE", "Game\xa0Date"] #nba.com uses different column names for the date in different tables

START_SEASON = 2006  # the first season to start scraping if full_scrape is TRUE (some seasons prior to 2006 may be missing some of the additional stats)


def main(full_scrape: bool = False):
    """
    Main function to scrape the data from nba.com and save it to csv files for later processing

    The way the data on nba.com is structured, we need to scrape the data in chunks based on the season, sub-season (regular vs playoffs),
    and the type of stats we want to scrape.
    
    Args:
        full_scrape (bool): whether to scrape all seasons from START_SEASON to current season, or just the most recent games
        
    """

    # determine when to start scraping 
    if full_scrape:
        # set start date to the beginning of the first season we want to scrape
        seasons = list(range(START_SEASON, datetime.now().year))
        seasons = [str(season) + "-" + (str(season + 1))[-2:] for season in seasons]  # format season as '2006-07' which is required for nba.com advanced boxscores
        first_start_date = str(REGULAR_SEASON_START)+ "/1/" + str(START_SEASON)  #usually starts in October
    else:
        # determine start date by looking at the cumulative scraped data
        
        scraped_data = [] #list of dataframes
        scraped_data = load_scraped_data(cumulative=True) # we want the cumulative scraped data
        
        # check the latest game in the dataset to see when to start scraping
        first_start_date, seasons = determine_scrape_start(scraped_data)

        if first_start_date is None:
            print("Error - previous scraped data has inconsistent dates")
            exit()

    # activate the web driver
    driver = activate_web_driver("Chrome")

    # scrape the data, cycling through the seasons and stat types
    for stat_type in STAT_TYPES:
        
        new_games = pd.DataFrame()
        df_season = pd.DataFrame()
        
        # if there are multiple seasons, start date will be reset to the beginning of the next season,
        # so we need to keep track of the original start date because we are scraping multiple stats categories, each with their own seasons
        start_date = first_start_date 

        for season in seasons:
            season_year = int(season[:4])    
            end_date = str(OFF_SEASON_START) + "/01/" + str(season_year+1)  # use start of off-season as end date for regular season
            df_season = scrape_sub_seasons(driver, str(season), str(start_date), str(end_date), stat_type)
            new_games = pd.concat([new_games, df_season], axis=0)
            start_date = str(REGULAR_SEASON_START) + "/01/" + str(season_year+1)  #if more than 1 season, reset start date to beginning of next season

        file_name = "games_" + stat_type + ".csv"
        save_scraped_data(new_games, file_name)
        

    # scrape the matchups (team ids) and games (game ids) for games scheduled today and save to csv
    scrape_and_save_todays_matchups(driver)

    #close web driver now that we are done scraping
    driver.close() 

    # reload the csv files and check that the data is consistent - same games in each dataframe, same number of rows, no nulls
    scraped_data = load_scraped_data(cumulative=False)

    response = validate_scraped_dataframes(scraped_data)

    if response == "Pass":
        print("All scraped dataframes are consistent")
    else:
        print("Error - scraped dataframes are inconsistent")
        print(response)



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
        #"--window-size=1920,1200",
        #"--start-maximized",
        #"--no-sandbox",
        #"--disable-dev-shm-usage",
        #"--disable-gpu",
        #"--ignore-certificate-errors",
        #"--disable-extensions",
        #"--disable-popup-blocking",
        #"--disable-notifications",
        "--remote-debugging-port=9222", #https://stackoverflow.com/questions/56637973/how-to-fix-selenium-devtoolsactiveport-file-doesnt-exist-exception-in-python
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
        #"--disable-blink-features=AutomationControlled",
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
        
        #check if the date column is a recognized column name and convert to datetime so we can find the latest date
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

    # determine the season for that date (the season is the year the season started, NBA starts in the fall)
    if last_date.month >= 10:
        last_season = last_date.year
    else:
        last_season = last_date.year - 1

    # Determine the date of the next day to begin scraping from
    start_date = last_date + timedelta(days=1)
    # format data to match nba.com boxscore url (mm/dd/yyyy)

    start_date = start_date.strftime("%m/%d/%Y")

    # determine what season we are in currently
    today = datetime.now(timezone('EST')) #nba.com uses US Eastern Standard Time
    if today.month >= 10:
        current_season = today.year
    else:
        current_season = today.year - 1

    # determine which seasons we need to scrape to catch up the data
    seasons = list(range(last_season, current_season+1))
    seasons = [str(season) + "-" + (str(season + 1))[-2:] for season in seasons]  # format season as '2005-06' which is required for nba.com advanced boxscores

    print("Last date in dataset: ", last_date)
    print("Last season in dataset: ", last_season)
    print("Current season: ", current_season)
    print("Seasons to scrape: ", seasons)
    print("Start date: ", start_date)

    return start_date, seasons


def scrape_sub_seasons(driver: webdriver, season: str, start_date: str, end_date: str, stat_type: str)-> pd.DataFrame:
    """
    Scrape each of the sub-seasons (Regular Season, Play-In, Playoffs) from nba.com and combine into a single DataFrame

    Args:
        driver (webdriver): selenium webdriver
        season (str): season to scrape, an empty string will default to the current season
        start_date (str): start date to scrape
        end_date (str): end date to scrape
        stat_type (str): type of stats to scrape - nba.com serves 5 different boxscore screens - traditional, advanced, four-factors, misc, scoring

    Returns:
        pd.DataFrame: the updated DataFrame
    """ 

    # nba website has 4 types of "sub-seasons" that are split onto different pages
    # Preseason, Regular Season, Play-In, Playoffs
    # (we will ignore Preseason because I don't think it is useful for our purposes)
    # we will scrape each of these as necessary and combine into a single DataFrame

    # if the dates are out of range for the sub-season (e.g. trying to use a May start date for the Regular Season), 
    # then instead of serving an empty table, the NBA website defaults to serving the entire sub-season's data with all the pages that encompasses
    # to minimize the number of scrape attempts and redundant data, we will TRY to only scrape the sub-seasons that align with our date range
    # if the date range includes April (the transition month), then we will probably end up scraping a lot of redundant data that will be filtered out later
    # (there might be a better way to do this, but I am going to brute force it for now)

    print(f"Scraping {season} from {start_date} to {end_date} for {stat_type} stats")
             
    all_sub_seasons = pd.DataFrame()

    for sub_season_type in SUB_SEASON_TYPES:
        
        df = scrape_to_dataframe(driver=driver, Season=season, DateFrom=start_date, DateTo=end_date, stat_type=stat_type, season_type=sub_season_type)

        if not(df.empty):
            all_sub_seasons = pd.concat([all_sub_seasons, df], axis=0)

    return all_sub_seasons


def validate_scraped_dataframes(scraped_dataframes: list) -> str:
    """
    Check and make sure that the scraping of all the dataframes went well and that each dataframe has the same games

    Args:
        scraped_dataframes (list): the list of scraped dataframes

    Returns:
        response (str): a string indicating if the dataframes are valid or not

    """

    response = "Pass"
    num_rows = 0
    game_ids = None
    
    # check for duplicate records
    for i, df in enumerate(scraped_dataframes):
        if df.duplicated().any():
            response = (f"Dataframe {i} has duplicate records")
            return response
        
    # check for nulls. We will handle "-" values later in the data processing, but now we are checking that the scraping went well
    for i, df in enumerate(scraped_dataframes):
        if df.isnull().values.any():
            response = (f"Dataframe {i} has null values")
            return False

    # verify that all the dataframes have the same games
    for i, df in enumerate(scraped_dataframes):
        
        # sort the dataframes by GAME_ID
        df = df.sort_values(by='GAME_ID')

        if i == 0:
            num_rows = df.shape[0]
            game_ids = df['GAME_ID']
        else:
            if num_rows != df.shape[0]:
                response = (f"Dataframe {i} does not match the number of rows of the first dataframe")
                print(num_rows, len(df))
                print(response)
            
            # check if the GAME_IDs match, use numpy array_equal to avoid issues with different index values
            if not np.array_equal(game_ids.values, df['GAME_ID'].values):
                response = (f"Dataframe {i} does not match the game ids of the first dataframe")
                print(response)
            
    return response


def parse_ids(data_table: soup) -> tuple[pd.Series, pd.Series]:
    """
    Parse the html table to extract the team and game ids

    Args:
        data_table (soup): html table from nba.com boxscores page

    Returns:
        pd.Series of game ids and team ids
    """
    
    # TEAM_ID and GAME_ID are encoded in href= links
    # find all the hrefs, add them to a list
    # then parse out a list for teams ids and game ids
    # and convert these to pandas series
    
    CLASS_ID = 'Anchor_anchor__cSc3P' 

    # get all the links
    links = data_table.find_all('a', {'class':CLASS_ID})
    
    # get the href part (web addresses)
    # href="/stats/team/1610612740" for teams
    # href="/game/0022200191" for games
    links_list = [i.get("href") for i in links]

    # create a series using last 10 digits of the appropriate links
    team_id = pd.Series([i[-10:] for i in links_list if ('stats' in i)])
    game_id = pd.Series([i[-10:] for i in links_list if ('/game/' in i)])
    
    return team_id, game_id



def scrape_to_dataframe(driver: webdriver, Season: str, DateFrom: str ="NONE", DateTo: str ="NONE", stat_type: str ='traditional', season_type: str = "Regular+Season") -> pd.DataFrame:
    """
    Retrieves stats from nba.com and converts to a DataFrame

    Args:
        driver (webdriver): selenium webdriver, is None if using scrapingant
        Season (str): season to scrape, is empty str if using current season
        DateFrom (str, optional): start date to scrape. Defaults to "NONE".
        DateTo (str, optional): end date to scrape. Defaults to "NONE".
        stat_type (str, optional): type of stats to scrape. Defaults to 'traditional'.
        season_type (str, optional): type of season to scrape. Defaults to "Regular+Season".

    Returns:
        pd.DataFrame: scraped DataFrame

    """
    # go to boxscores webpage at nba.com
    # check if the data table is split over multiple pages 
    # if so, then select the "ALL" choice in pulldown menu to show all on one page
    # extract out the html table and convert to dataframe
    # parse out GAME_ID and TEAM_ID from href links
    # and add these to DataFrame

    # originally I started using beautifulsoup for this project,
    # but I switched to selenium because the nba.com website is dynamic uses a lot of javascript
    # so currently there is a mix of both in this function
    
    # if season not provided, then will default to current season
    # if DateFrom and DateTo not provided, then don't include in url - pull the whole season

    # key classes for scraping determined by visual inspection of page source code
    CLASS_ID_TABLE = 'Crom_table__p1iZz' 
    CLASS_ID_PAGINATION = "Pagination_pageDropdown__KgjBU" 
    CLASS_ID_DROPDOWN = "DropDown_select__4pIg9" 

    
    if stat_type == 'traditional':
        nba_url = NBA_BOXSCORES + "?SeasonType=" + season_type
    else:
        nba_url = NBA_BOXSCORES + "-"+ stat_type + "?SeasonType=" + season_type
        
    if not Season:
        nba_url = nba_url + "&DateFrom=" + DateFrom + "&DateTo=" + DateTo
    else:
        if DateFrom == "NONE" and DateTo == "NONE":
            nba_url = nba_url + "&Season=" + Season
        else:
            nba_url = nba_url + "&Season=" + Season + "&DateFrom=" + DateFrom + "&DateTo=" + DateTo

    print(f"Scraping {nba_url}")

        
    driver.get(nba_url)
    time.sleep(10)
    source = soup(driver.page_source, 'html.parser')

    #check for more than one page
    pagination = source.find('div', {'class':CLASS_ID_PAGINATION})
    #pagination = driver.find_elements(By.XPATH, "//*[@class='" + CLASS_ID_PAGINATION + "']")

    if pagination is None:
        
        # if only one page, then just extract the data table
        
        data_table = source.find('table', {'class':CLASS_ID_TABLE})
        if data_table is None:
            print("No data found")
            return pd.DataFrame()
        
    else:
        
        # if multiple pages, first activate pulldown option for All pages to show all rows on one page
        
        page_dropdown = driver.find_element(By.XPATH, "//*[@class='" + CLASS_ID_PAGINATION + "']//*[@class='" + CLASS_ID_DROPDOWN + "']")
    
        page_dropdown.send_keys("ALL") # show all pages
        #page_dropdown.click() doesn't work in headless mode
        time.sleep(3)
        driver.execute_script('arguments[0].click()', page_dropdown) #click() didn't work in headless mode, used this workaround (https://stackoverflow.com/questions/57741875)
        
        #refresh page data now that it contains all rows of the table
        time.sleep(3)
        source = soup(driver.page_source, 'html.parser')
        data_table = source.find('table', {'class':CLASS_ID_TABLE})
                
        if data_table is None:
            print("Error finding data table")
            return pd.DataFrame()


    # convert the html table to a dataframe   

    dfs = pd.read_html(str(data_table), header=0) 
    df = pd.concat(dfs)

    # pull out teams ids and game ids from hrefs and add these to the dataframe
    TEAM_ID, GAME_ID = parse_ids(data_table)
    df['TEAM_ID'] = TEAM_ID
    df['GAME_ID'] = GAME_ID
        
    return df



def scrape_and_save_todays_matchups(driver: webdriver) -> None:
    """
    Scrapes the the teams playing today from the nba.com schedule page along with the game ids and saves to csv
    
    (There might be other websites where it is easier to scrape today's games, since NBA.com uses
    inconsistent layouts, but there is a certain convenience in also getting the correct NBA.com game ids 
    for the games. The lack of game ids from other sites could be worked around by finding them later
    after the game is complete from the boxscore page, but this is the way I chose to do it initially)

    Args:
        driver (webdriver): selenium driver

    """

    CLASS_GAMES_PER_DAY = "ScheduleDay_sdGames__NGdO5" # the div containing all games for a day
    CLASS_DAY = "ScheduleDay_sdDay__3s2Xt" # the heading with the date for the games (e.g. "Wednesday, February 1")
    
    # go to nba.com schedule page
    driver.get(NBA_SCHEDULE)
    time.sleep(10)
    source = soup(driver.page_source, 'html.parser')

    # Get the block of all of todays games
    # Sometimes, the results of yesterday's games are listed first, then todays games are listed
    # Other times, yesterday's games are not listed, or when the playoffs approach, future games are listed
    # We will check the date for the first div, if it is not todays date, then we will look for the next div

    div_games = source.find('div', {'class':CLASS_GAMES_PER_DAY}) # first div may or may not be yesterday's games or even future games when playoffs approach
    div_game_day = source.find('h4', {'class':CLASS_DAY})
    today = datetime.today().strftime('%A, %B %d')[:3] # e.g. "Wednesday, February 1" -> "Wed" for convenience with dealing with leading zeros
    todays_games = None
    
    while div_games:
        print('Found games for: ' + div_game_day.text[:3]) 
        if today == div_game_day.text[:3]:  
            todays_games = div_games
            break
        else:
            # move to next div
            div_games = div_games.find_next('div', {'class':CLASS_GAMES_PER_DAY}) 
            div_game_day = div_game_day.find_next('h4', {'class':CLASS_DAY})

    if todays_games is None:
        print("No games found for today")
        return [], []   

    # Get the teams playing
    # Each team listed in todays block will have a href with the specified anchor class
    # e.g. <a href="/team/1610612743/nuggets/" class="Anchor_anchor__cSc3P Link_styled__okbXW" ...
    # href includes team ID (1610612743 in example)
    # first team is visitor, second team is home
    CLASS_ID = "Anchor_anchor__cSc3P Link_styled__okbXW"
    links = todays_games.find_all('a', {'class':CLASS_ID})
    teams_list = [i.get("href") for i in links]

    # example output:
    # ['/team/1610612759/spurs/', '/team/1610612748/heat/',...

    # create list of matchups by parsing out team ids from teams_list
    # second team id is always the home team
    team_count = len(teams_list) 
    matchups = []
    for i in range(0,team_count,2):
        visitor_id = teams_list[i].partition("team/")[2].partition("/")[0] #extract team id from text
        home_id = teams_list[i+1].partition("team/")[2].partition("/")[0]
        matchups.append([visitor_id, home_id])


    # Get Game IDs
    # Each game listed in todays block will have a link with the specified anchor class
    # <a class="Anchor_anchor__cSc3P TabLink_link__f_15h" data-content="SAC @ MEM, 2023-01-01" data-content-id="0022200547" data-has-children="true" data-has-more="false" data-id="nba:schedule:main:preview:cta" data-is-external="false" data-text="PREVIEW" data-track="click" data-type="cta" href="/game/sac-vs-mem-0022200547">PREVIEW</a>
    # Each game will have two links with the specified anchor class, one for the preview and one to buy tickets
    # all using the same anchor class, so we will filter out those just for PREVIEW
    CLASS_ID = "Anchor_anchor__cSc3P TabLink_link__f_15h"
    links = todays_games.find_all('a', {'class':CLASS_ID})
    links = [i for i in links if "PREVIEW" in i]
    game_id_list = [i.get("href") for i in links]
    

    games = []
    for game in game_id_list:
        game_id = game.partition("-00")[2].partition("?")[0] # extract team id from text for link
        if len(game_id) > 0:               
            games.append(game_id)   

    # save the data to a csv file - will be empty if no games are scheduled for today
    matchups = pd.DataFrame(matchups)
    file_name = "matchups"
    save_scraped_data(matchups, file_name)
    games = pd.DataFrame(games)
    file_name = "games_ids"
    save_scraped_data(games, file_name)



if __name__ == "__main__":
    main()
    
