from datetime import datetime
from pathlib import Path

from .nba_scraper import NbaScraper
from .utils import (
    activate_web_driver, 
    get_start_date_and_seasons, 
    validate_data,
)


def main():
    """
    Runs the main functionality of the web scraping application.
    
    This function performs the following tasks:
    - Retrieves the start date and seasons to scrape data for
    - Activates the web driver for the Chrome browser
    - Initializes an NbaScraper instance and uses it to:
        - Scrape and save all box scores for the specified seasons
        - Scrape and save the matchups for today's date
    - Closes the web driver
    - Validates the scraped data
    """
        

    first_start_date, seasons = get_start_date_and_seasons()

    driver = activate_web_driver("Chrome")

    with NbaScraper(driver) as scraper:
        
        # scrape all boxscores for all seasons
        scraper.scrape_and_save_all_boxscores(seasons, first_start_date)

        # scrape schedule for today's matchups
        search_day = datetime.today().strftime('%A, %B %d')[:3]
        scraper.scrape_and_save_matchups_for_day(search_day)

    driver.close() 

    validate_data()

if __name__ == "__main__":
    main()