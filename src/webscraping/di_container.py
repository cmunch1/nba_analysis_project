from dependency_injector import containers, providers

from .concrete_classes import (
  WebDriver, 
  DataAccess, 
  PageScraper, 
  BoxscoreScraper, 
  ScheduleScraper, 
  NbaScraper,
)


class DIContainer(containers.DeclarativeContainer):
  web_driver = providers.Factory(WebDriver)
  data_access = providers.Factory(DataAccess)
  page_scraper = providers.Factory(PageScraper, web_driver=web_driver)
  boxscore_scraper = providers.Factory(BoxscoreScraper, data_access=data_access, page_scraper=page_scraper)
  schedule_scraper = providers.Factory(ScheduleScraper, data_access=data_access, page_scraper=page_scraper)
  nba_scraper = providers.Factory(NbaScraper, boxscore_scraper=boxscore_scraper, schedule_scraper=schedule_scraper)