from .nba_scraper import NbaScraper
from .utils import activate_web_driver, determine_scrape_start
from .validate import validate_scraped_dataframes
from ..data_access.data_access import load_scraped_data, save_scraped_data
from ..configs.configs import START_SEASON, REGULAR_SEASON_START

def main(full_scrape: bool = False):
    if full_scrape:
        seasons = list(range(START_SEASON, datetime.now().year))
        seasons = [str(season) + "-" + (str(season + 1))[-2:] for season in seasons]
        first_start_date = f"{REGULAR_SEASON_START}/1/{START_SEASON}"
    else:
        scraped_data = load_scraped_data(cumulative=True)
        first_start_date, seasons = determine_scrape_start(scraped_data)

        if first_start_date is None:
            print("Error - previous scraped data has inconsistent dates")
            exit()

    driver = activate_web_driver("Chrome")
    scraper = NbaScraper(driver)

    for stat_type in scraper.STAT_TYPES:
        new_games = scraper.scrape_stat_type(seasons, first_start_date, stat_type)
        file_name = f"games_{stat_type}.csv"
        save_scraped_data(new_games, file_name)

    search_day = datetime.today().strftime('%A, %B %d')[:3]
    scraper.scrape_and_save_matchups_for_day(search_day)

    driver.close() 

    scraped_data = load_scraped_data(cumulative=False)
    response = validate_scraped_dataframes(scraped_data)

    if response == "Pass":
        print("All scraped dataframes are consistent")
    else:
        print("Error - scraped dataframes are inconsistent")
        print(response)

if __name__ == "__main__":
    main()