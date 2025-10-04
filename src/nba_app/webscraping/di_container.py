from dependency_injector import containers, providers

from platform_core.core.common_di_container import CommonDIContainer
from ..data_validator import DataValidator

from .web_driver import CustomWebDriver
from .page_scraper import PageScraper 
from .boxscore_scraper import BoxscoreScraper 
from .schedule_scraper import ScheduleScraper 
from .nba_scraper import NbaScraper
from .matchup_validator import MatchupValidator


class DIContainer(CommonDIContainer):
    """
    Webscraping-specific DI container that inherits from CommonDIContainer.
    Adds webscraping-specific dependencies while reusing common ones.
    """
    
    # Override data_validator with the specific implementation for this app
    data_validator = providers.Factory(
        DataValidator,
        config=CommonDIContainer.config,
        data_access=CommonDIContainer.data_access,
        app_logger=CommonDIContainer.app_logger,
        app_file_handler=CommonDIContainer.app_file_handler,
        error_handler=CommonDIContainer.error_handler_factory
    )
    
    # Webscraping-specific dependencies
    web_driver_factory = providers.Singleton(CustomWebDriver, config=CommonDIContainer.config, app_logger=CommonDIContainer.app_logger)
    driver = providers.Singleton(
        lambda web_driver_factory: web_driver_factory.create_driver(),
        web_driver_factory=web_driver_factory
    )

    page_scraper = providers.Factory(
        PageScraper,
        config=CommonDIContainer.config,
        web_driver=driver,
        app_logger=CommonDIContainer.app_logger,
        error_handler=CommonDIContainer.error_handler_factory
    )

    boxscore_scraper = providers.Factory(
        BoxscoreScraper,
        config=CommonDIContainer.config,
        data_access=CommonDIContainer.data_access,
        page_scraper=page_scraper,
        app_logger=CommonDIContainer.app_logger,
        error_handler=CommonDIContainer.error_handler_factory
    )

    schedule_scraper = providers.Factory(
        ScheduleScraper,
        config=CommonDIContainer.config,
        data_access=CommonDIContainer.data_access,
        page_scraper=page_scraper,
        app_logger=CommonDIContainer.app_logger,
        error_handler=CommonDIContainer.error_handler_factory
    )

    nba_scraper = providers.Factory(
        NbaScraper,
        config=CommonDIContainer.config,
        boxscore_scraper=boxscore_scraper,
        schedule_scraper=schedule_scraper,
        app_logger=CommonDIContainer.app_logger,
        error_handler=CommonDIContainer.error_handler_factory
    )

    matchup_validator = providers.Factory(
        MatchupValidator,
        config=CommonDIContainer.config,
        data_access=CommonDIContainer.data_access,
        page_scraper=page_scraper,
        app_logger=CommonDIContainer.app_logger,
        error_handler=CommonDIContainer.error_handler_factory
    )
