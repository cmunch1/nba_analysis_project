from dependency_injector import containers, providers

from ..config.config import Config  

from .concrete_classes import (
  WebDriver_, 
  DataAccess, 
  PageScraper, 
  BoxscoreScraper, 
  ScheduleScraper, 
  NbaScraper,
)

class DIContainer(containers.DeclarativeContainer):
  config = providers.Singleton(Config)  # Create a singleton instance of Config
  
  web_driver = providers.Factory(WebDriver_, config=config)
  data_access = providers.Factory(DataAccess, config=config)
  page_scraper = providers.Factory(PageScraper, config=config, web_driver=web_driver)
  boxscore_scraper = providers.Factory(BoxscoreScraper, config=config, data_access=data_access, page_scraper=page_scraper)
  schedule_scraper = providers.Factory(ScheduleScraper, config=config, data_access=data_access, page_scraper=page_scraper)
  nba_scraper = providers.Factory(NbaScraper, config=config, boxscore_scraper=boxscore_scraper, schedule_scraper=schedule_scraper)