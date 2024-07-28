from datetime import datetime
from pathlib import Path

from .nba_scraper import NbaScraper
from .utils import (
    activate_web_driver, 
    get_start_date_and_seasons, 
    validate_data,
)


def main():
    

    first_start_date, seasons = get_start_date_and_seasons()

    driver = activate_web_driver("Chrome")

    scraper = NbaScraper(driver)

    # scrape boxscores for each stat type
    scraper.scrape_and_save_all_boxscores(seasons, first_start_date)

    # scrape schedule for today's matchups
    search_day = datetime.today().strftime('%A, %B %d')[:3]
    scraper.scrape_and_save_matchups_for_day(search_day)

    driver.close() 

    validate_data()

if __name__ == "__main__":
    main()