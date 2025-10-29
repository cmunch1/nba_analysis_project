"""Dependency injection container for Streamlit dashboard."""

from dependency_injector import containers, providers

from ml_framework.core.common_di_container import CommonDIContainer
from streamlit_app.services.dashboard_data_service import DashboardDataService


class StreamlitAppContainer(CommonDIContainer):
    """
    Compose core dependencies with Streamlit-specific services.
    """

    dashboard_data_service = providers.Factory(
        DashboardDataService,
        config=CommonDIContainer.config,
        app_logger=CommonDIContainer.app_logger,
        app_file_handler=CommonDIContainer.app_file_handler,
        error_handler=CommonDIContainer.error_handler_factory,
    )
