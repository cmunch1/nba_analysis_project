import pandas as pd
from selenium.webdriver.remote.webelement import WebElement

from .config import config
from .page_scraper import PageScraper


from ..data_access.data_access import save_scraped_data


class NbaScraper:
    def __init__(self, driver):
        self.driver = driver
        self.page_scraper = PageScraper(driver)

        # Load configurations
                
        # season info
        self.start_season = config["start_season"]   #season to start scraping if choosing a full scrape, more advanced stats don't go back beyond 2006
        self.regular_season_start_month = config["regular_season_start_month"]
        self.off_season_start_month = config["off_season_start_month"]
        # boxscores sub-pages - url construction requires that these be specified
        self.sub_season_types = config["sub_season_types"] #["Regular+Season", "PlayIn", "Playoffs"],
        self.stat_types = config["stat_types"] #["traditional", "advanced", "four-factors", "misc", "scoring"],
        # boxscores
        self.nba_boxscores_url = config["nba_boxscores_url"]
        self.table_class_name = config["table_class_name"]  
        self.pagination_class_name = config["pagination_class_name"]  
        self.dropdown_class_name = config["dropdown_class_name"]
        self.teams_and_games_class_name = config["teams_and_games_class_name"]
        # schedule
        self.nba_schedule_url = config["nba_schedule_url"]
        self.games_per_day_class_name = config["games_per_day_class_name"]
        self.day_class_name = config["day_class_name"]
        self.teams_links_class_name = config["teams_links_class_name"]
        self.game_links_class_name = config["game_links_class_name"] 


    def scrape_sub_seasons(self, season: str, start_date: str, end_date: str, stat_type: str) -> pd.DataFrame:
        print(f"Scraping {season} from {start_date} to {end_date} for {stat_type} stats")
        
        all_sub_seasons = pd.DataFrame()

        for sub_season_type in self.sub_season_types:
            df = self.scrape_to_dataframe(Season=season, DateFrom=start_date, DateTo=end_date, stat_type=stat_type, season_type=sub_season_type)
            if not df.empty:
                all_sub_seasons = pd.concat([all_sub_seasons, df], axis=0)

        return all_sub_seasons

    def scrape_boxscores_table(self, Season: str, DateFrom: str ="NONE", DateTo: str ="NONE", stat_type: str ='traditional', season_type: str = "Regular+Season") -> WebElement:

        nba_url = self.construct_nba_url(stat_type, season_type, Season, DateFrom, DateTo)
        print(f"Scraping {nba_url}")

        data_table = self.page_scraper.scrape_page_table(nba_url, self.table_class_name, self.pagination_class_name, self.dropdown_class_name)
        
        return data_table
    
    def convert_table_to_df(self, data_table):
        table_html = data_table.get_attribute('outerHTML')
        dfs = pd.read_html(table_html, header=0)
        df = pd.concat(dfs)

        team_id, game_id = self.extract_team_and_game_ids_boxscores(data_table)
        df['TEAM_ID'] = team_id
        df['GAME_ID'] = game_id

        return df

    def construct_nba_url(self, stat_type, season_type, Season, DateFrom, DateTo):
        if stat_type == 'traditional':
            nba_url = f"{self.nba_boxscores_url}?SeasonType={season_type}"
        else:
            nba_url = f"{self.nba_boxscores_url}-{stat_type}?SeasonType={season_type}"
            
        if not Season:
            nba_url = f"{nba_url}&DateFrom={DateFrom}&DateTo={DateTo}"
        else:
            if DateFrom == "NONE" and DateTo == "NONE":
                nba_url = f"{nba_url}&Season={Season}"
            else:
                nba_url = f"{nba_url}&Season={Season}&DateFrom={DateFrom}&DateTo={DateTo}"
        return nba_url

    def scrape_and_save_matchups_for_day(self, search_day) -> None:
  
        self.page_scraper.go_to_url(self.nba_schedule_url)

        days_games = self.find_games_for_day(search_day)
        
        if days_games is None:
            print("No games found for this day")
            return

        matchups = self.extract_team_ids(days_games)
        games = self.extract_game_ids(days_games)

        self.save_matchups_and_games(matchups, games)

    def find_games_for_day(self, search_day):
        
        game_days = self.page_scraper.get_elements_by_class(self.day_class_name)
        games_containers = self.page_scraper.get_elements_by_class(self.games_per_day_class_name)
        
        for day, days_games in zip(game_days, games_containers):
            if search_day == day.text[:3]:
                return days_games
        return None

    def extract_team_ids_schedule(self, todays_games):
        links = self.page_scraper.get_elements_by_class(self.teams_links_class_name, todays_games)
        teams_list = [i.get_attribute("href") for i in links]

        matchups = []
        for i in range(0, len(teams_list), 2):
            visitor_id = teams_list[i].partition("team/")[2].partition("/")[0]
            home_id = teams_list[i+1].partition("team/")[2].partition("/")[0]
            matchups.append([visitor_id, home_id])
        return matchups

    def extract_game_ids_schedule(self, todays_games):
        links = self.page_scraper.get_elements_by_class(self.game_links_class_name, todays_games)
        links = [i for i in links if "PREVIEW" in i.text]
        game_id_list = [i.get_attribute("href") for i in links]
        
        games = []
        for game in game_id_list:
            game_id = game.partition("-00")[2].partition("?")[0]
            if len(game_id) > 0:               
                games.append(game_id)
        return games
    
    def extract_team_and_game_ids_boxscores(self, data_table):
        links = self.page_scraper.get_elements_by_class(self.teams_and_games_class_name, data_table)       
        links_list = [i.get_attribute("href") for i in links]
    
        # href="/stats/team/1610612740" for teams
        # href="/game/0022200191" for games 

        # create a series using last 10 digits of the appropriate links
        team_id = pd.Series([i[-10:] for i in links_list if ('stats' in i)])
        game_id = pd.Series([i[-10:] for i in links_list if ('/game/' in i)])
    
        return team_id, game_id

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
            end_date = f"{self.off_season_start_month}/01/{season_year+1}"
            df_season = self.scrape_sub_seasons(str(season), str(start_date), str(end_date), stat_type)
            new_games = pd.concat([new_games, df_season], axis=0)
            start_date = f"{self.regular_season_start_month}/01/{season_year+1}"

        return new_games

    def scrape_and_save_all_boxscores(self, seasons, first_start_date):

        for stat_type in self.stat_types:
            new_games = self.scrape_stat_type(seasons, first_start_date, stat_type)
            file_name = f"games_{stat_type}.csv"
            save_scraped_data(new_games, file_name)
    
    

    
    