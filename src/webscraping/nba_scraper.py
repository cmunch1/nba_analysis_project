import pandas as pd
from datetime import datetime
import json


from .page_scraper import PageScraper
from ..data_access.data_access import save_scraped_data
from ..configs.configs import OFF_SEASON_START, REGULAR_SEASON_START


class NbaScraper:
    def __init__(self, driver):
        self.driver = driver
        self.page_scraper = PageScraper(driver)

         # Load configurations
        with open('config.json') as config_file:
            config = json.load(config_file)
        
        self.nba_boxscores_url = config["nba_boxscores_url"]
        self.nba_schedule_url = config["nba_schedule_url"]
        self.sub_season_types = config["sub_season_types"]
        self.stat_types = config["stat_types"]
        self.table_class_name = config["table_class_name"]  
        self.pagination_class_name = config["pagination_class_name"]  
        self.dropdown_class_name = config["dropdown_class_name"]
        self.games_per_day_class_name = config["games_per_day_class_name"]
        self.day_class_name = config["day_class_name"]
        self.teams_class_name = config["teams_class_name"]
        self.game_links_class_name = config["game_links_class_name"] 

    def scrape_sub_seasons(self, season: str, start_date: str, end_date: str, stat_type: str) -> pd.DataFrame:
        print(f"Scraping {season} from {start_date} to {end_date} for {stat_type} stats")
        
        all_sub_seasons = pd.DataFrame()

        for sub_season_type in self.sub_season_types:
            df = self.scrape_to_dataframe(Season=season, DateFrom=start_date, DateTo=end_date, stat_type=stat_type, season_type=sub_season_type)
            if not df.empty:
                all_sub_seasons = pd.concat([all_sub_seasons, df], axis=0)

        return all_sub_seasons

    def scrape_to_dataframe(self, Season: str, DateFrom: str ="NONE", DateTo: str ="NONE", stat_type: str ='traditional', season_type: str = "Regular+Season") -> pd.DataFrame:

        nba_url = self.construct_nba_url(stat_type, season_type, Season, DateFrom, DateTo)
        print(f"Scraping {nba_url}")

        df = self.page_scraper.scrape_page_table(nba_url, self.table_class_name, self.pagination_class_name, self.dropdown_class_name)
        
        return df

    def construct_nba_url(self, stat_type, season_type, Season, DateFrom, DateTo):
        if stat_type == 'traditional':
            nba_url = f"{self.NBA_BOXSCORES}?SeasonType={season_type}"
        else:
            nba_url = f"{self.NBA_BOXSCORES}-{stat_type}?SeasonType={season_type}"
            
        if not Season:
            nba_url = f"{nba_url}&DateFrom={DateFrom}&DateTo={DateTo}"
        else:
            if DateFrom == "NONE" and DateTo == "NONE":
                nba_url = f"{nba_url}&Season={Season}"
            else:
                nba_url = f"{nba_url}&Season={Season}&DateFrom={DateFrom}&DateTo={DateTo}"
        return nba_url

    def scrape_and_save_todays_matchups(self) -> None:

        
        self.page_scraper.go_to_url(self.NBA_SCHEDULE)

        todays_games = self.find_todays_games(CLASS_DAY, self.games_per_day_class_name)
        
        if todays_games is None:
            print("No games found for today")
            return

        matchups = self.extract_team_ids(todays_games)
        games = self.extract_game_ids(todays_games)

        self.save_matchups_and_games(matchups, games)

    def find_todays_games(self, day_class_name, self.games_per_day_class_name):
        today = datetime.today().strftime('%A, %B %d')[:3]
        game_days = self.page_scraper.get_elements_by_class(self.day_class_name)
        games_containers = self.page_scraper.get_elements_by_class(self.games_per_day_class_name)
        
        for day, games in zip(game_days, games_containers):
            if today == day.text[:3]:
                return games
        return None

    def extract_team_ids(self, todays_games):
        links = todays_games.find_elements(By.CLASS_NAME, self.teams_class_name)
        teams_list = [i.get_attribute("href") for i in links]

        matchups = []
        for i in range(0, len(teams_list), 2):
            visitor_id = teams_list[i].partition("team/")[2].partition("/")[0]
            home_id = teams_list[i+1].partition("team/")[2].partition("/")[0]
            matchups.append([visitor_id, home_id])
        return matchups

    def extract_game_ids(self, todays_games):
        links = todays_games.find_elements(By.CLASS_NAME, self.game_links_class_name)
        links = [i for i in links if "PREVIEW" in i.text]
        game_id_list = [i.get_attribute("href") for i in links]
        
        games = []
        for game in game_id_list:
            game_id = game.partition("-00")[2].partition("?")[0]
            if len(game_id) > 0:               
                games.append(game_id)
        return games

    def save_matchups_and_games(self, matchups, games):
        matchups_df = pd.DataFrame(matchups)
        save_scraped_data(matchups_df, "matchups")
        games_df = pd.DataFrame(games)
        save_scraped_data(games_df, "games_ids")

    def scrape_stat_type(self, seasons, first_start_date, stat_type):
        new_games = pd.DataFrame()
        start_date = first_start_date

        for season in seasons:
            season_year = int(season[:4])    
            end_date = f"{OFF_SEASON_START}/01/{season_year+1}"
            df_season = self.scrape_sub_seasons(str(season), str(start_date), str(end_date), stat_type)
            new_games = pd.concat([new_games, df_season], axis=0)
            start_date = f"{REGULAR_SEASON_START}/01/{season_year+1}"

        return new_games
    
    