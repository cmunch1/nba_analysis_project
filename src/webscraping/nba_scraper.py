import pandas as pd
from datetime import datetime


from .scraped_page import ScrapedPage
from ..data_access.data_access import save_scraped_data
from ..configs.configs import OFF_SEASON_START, REGULAR_SEASON_START

class NbaScraper:
    def __init__(self, driver):
        self.driver = driver
        self.scraped_page = ScrapedPage(driver)
        self.NBA_BOXSCORES = "https://www.nba.com/stats/teams/boxscores"
        self.NBA_SCHEDULE = "https://www.nba.com/schedule"
        self.SUB_SEASON_TYPES = ["Regular+Season", "PlayIn", "Playoffs"]
        self.STAT_TYPES = ['traditional', 'advanced', 'four-factors', 'misc', 'scoring']

    def scrape_sub_seasons(self, season: str, start_date: str, end_date: str, stat_type: str) -> pd.DataFrame:
        print(f"Scraping {season} from {start_date} to {end_date} for {stat_type} stats")
        
        all_sub_seasons = pd.DataFrame()

        for sub_season_type in self.SUB_SEASON_TYPES:
            df = self.scrape_to_dataframe(Season=season, DateFrom=start_date, DateTo=end_date, stat_type=stat_type, season_type=sub_season_type)
            if not df.empty:
                all_sub_seasons = pd.concat([all_sub_seasons, df], axis=0)

        return all_sub_seasons

    def scrape_to_dataframe(self, Season: str, DateFrom: str ="NONE", DateTo: str ="NONE", stat_type: str ='traditional', season_type: str = "Regular+Season") -> pd.DataFrame:
        CLASS_ID_TABLE = 'Crom_table__p1iZz' 
        CLASS_ID_PAGINATION = "Pagination_pageDropdown__KgjBU" 
        CLASS_ID_DROPDOWN = "DropDown_select__4pIg9" 

        nba_url = self.construct_nba_url(stat_type, season_type, Season, DateFrom, DateTo)
        print(f"Scraping {nba_url}")

        df = self.scraped_page.scrape_page(nba_url, CLASS_ID_TABLE, CLASS_ID_PAGINATION, CLASS_ID_DROPDOWN)
        
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
        CLASS_GAMES_PER_DAY = "ScheduleDay_sdGames__NGdO5"
        CLASS_DAY = "ScheduleDay_sdDay__3s2Xt"
        
        self.scraped_page.go_to_url(self.NBA_SCHEDULE)

        try:
            self.scraped_page.wait.until(EC.presence_of_element_located((By.CLASS_NAME, CLASS_GAMES_PER_DAY)))
        except TimeoutException:
            print("No games found for today")
            return

        todays_games = self.find_todays_games(CLASS_DAY, CLASS_GAMES_PER_DAY)
        
        if todays_games is None:
            print("No games found for today")
            return

        matchups = self.extract_team_ids(todays_games)
        games = self.extract_game_ids(todays_games)

        self.save_matchups_and_games(matchups, games)

    def find_todays_games(self, CLASS_DAY, CLASS_GAMES_PER_DAY):
        today = datetime.today().strftime('%A, %B %d')[:3]
        game_days = self.scraped_page.get_elements_by_class(CLASS_DAY)
        games_containers = self.scraped_page.get_elements_by_class(CLASS_GAMES_PER_DAY)
        
        for day, games in zip(game_days, games_containers):
            if today == day.text[:3]:
                return games
        return None

    def extract_team_ids(self, todays_games):
        CLASS_ID = "Anchor_anchor__cSc3P Link_styled__okbXW"
        links = todays_games.find_elements(By.CLASS_NAME, CLASS_ID)
        teams_list = [i.get_attribute("href") for i in links]

        matchups = []
        for i in range(0, len(teams_list), 2):
            visitor_id = teams_list[i].partition("team/")[2].partition("/")[0]
            home_id = teams_list[i+1].partition("team/")[2].partition("/")[0]
            matchups.append([visitor_id, home_id])
        return matchups

    def extract_game_ids(self, todays_games):
        CLASS_ID = "Anchor_anchor__cSc3P TabLink_link__f_15h"
        links = todays_games.find_elements(By.CLASS_NAME, CLASS_ID)
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
    
    