from datetime import datetime
from pathlib import Path

from .nba_scraper import NbaScraper
from .utils import (
    get_start_date_and_seasons, 
    validate_data,
    concatenate_scraped_data,
)

from ..logging.logging_setup import setup_logging
setup_logging("webscraping.log")

def main():
    """
    Runs the main functionality of the web scraping application.
    
    This function performs the following tasks:
    - Retrieves the start date and seasons to scrape data for
    - Activates the web driver for the Chrome browser
    - Initializes an NbaScraper instance and uses it to:
        - Scrape and save all box scores for the specified seasons
        - Scrape and save the matchups for today's date
    - Validates the scraped data
    """

    first_start_date, seasons = get_start_date_and_seasons()


    with NbaScraper() as scraper:
        
        # scrape all boxscores for all seasons
        scraper.scrape_and_save_all_boxscores(seasons, first_start_date)

        # scrape schedule for today's matchups
        search_day = datetime.today().strftime('%A, %B %d')[:3]
        scraper.scrape_and_save_matchups_for_day(search_day)

    
    validate_data(cumulative=False)

    # combine newly scraped data with cumulative scraped data
    concatenate_scraped_data()

    validate_data(cumulative=True)



if __name__ == "__main__":
    main()