from dependency_injector import containers, providers

from platform_core.core.config_management.config_manager import ConfigManager  
from platform_core.framework.data_access.csv_data_access import CSVDataAccess
from ..data_validation.data_validator import DataValidator

from .web_driver import CustomWebDriver
from .page_scraper import PageScraper 
from .boxscore_scraper import BoxscoreScraper 
from .schedule_scraper import ScheduleScraper 
from .nba_scraper import NbaScraper
from .matchup_validator import MatchupValidator


class DIContainer(containers.DeclarativeContainer):
    config = providers.Singleton(ConfigManager)
    
    web_driver_factory = providers.Singleton(CustomWebDriver, config=config)
    driver = providers.Singleton(
          lambda web_driver_factory: web_driver_factory.create_driver(),
          web_driver_factory=web_driver_factory
      )

    data_access = providers.Factory(CSVDataAccess, config=config)
    data_validator = providers.Factory(DataValidator, config=config)
    page_scraper = providers.Factory(PageScraper, config=config, web_driver=driver)
    boxscore_scraper = providers.Factory(BoxscoreScraper, config=config, data_access=data_access, page_scraper=page_scraper)
    schedule_scraper = providers.Factory(ScheduleScraper, config=config, data_access=data_access, page_scraper=page_scraper)
    nba_scraper = providers.Factory(NbaScraper, config=config, boxscore_scraper=boxscore_scraper, schedule_scraper=schedule_scraper)
    matchup_validator = providers.Factory(
        MatchupValidator,
        config=config,
        data_access=data_access,
        page_scraper=page_scraper
    )
 