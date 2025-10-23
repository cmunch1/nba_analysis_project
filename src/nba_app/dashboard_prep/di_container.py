"""Dependency Injection Container for Dashboard Prep Module"""

from dependency_injector import containers, providers

from ml_framework.core.common_di_container import CommonDIContainer
from .predictions_aggregator import PredictionsAggregator
from .results_aggregator import ResultsAggregator
from .performance_calculator import PerformanceCalculator
from .team_performance_analyzer import TeamPerformanceAnalyzer
from .dashboard_data_generator import DashboardDataGenerator


class DIContainer(CommonDIContainer):
    """
    Dashboard prep-specific DI container.

    Inherits common infrastructure and adds dashboard prep components.
    """

    # Predictions aggregator
    predictions_aggregator = providers.Factory(
        PredictionsAggregator,
        config=CommonDIContainer.config,
        app_logger=CommonDIContainer.app_logger,
        data_access=CommonDIContainer.data_access,
        error_handler=CommonDIContainer.error_handler_factory
    )

    # Results aggregator
    results_aggregator = providers.Factory(
        ResultsAggregator,
        config=CommonDIContainer.config,
        app_logger=CommonDIContainer.app_logger,
        data_access=CommonDIContainer.data_access,
        error_handler=CommonDIContainer.error_handler_factory
    )

    # Performance calculator
    performance_calculator = providers.Factory(
        PerformanceCalculator,
        config=CommonDIContainer.config,
        app_logger=CommonDIContainer.app_logger,
        error_handler=CommonDIContainer.error_handler_factory
    )

    # Team performance analyzer
    team_performance_analyzer = providers.Factory(
        TeamPerformanceAnalyzer,
        config=CommonDIContainer.config,
        app_logger=CommonDIContainer.app_logger,
        error_handler=CommonDIContainer.error_handler_factory
    )

    # Dashboard data generator (orchestrator)
    dashboard_data_generator = providers.Factory(
        DashboardDataGenerator,
        config=CommonDIContainer.config,
        app_logger=CommonDIContainer.app_logger,
        data_access=CommonDIContainer.data_access,
        error_handler=CommonDIContainer.error_handler_factory,
        predictions_aggregator=predictions_aggregator,
        results_aggregator=results_aggregator,
        performance_calculator=performance_calculator,
        team_performance_analyzer=team_performance_analyzer
    )
