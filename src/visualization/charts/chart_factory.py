from typing import Dict, Type
from .chart_types import ChartType
from .base_chart import BaseChart
from .feature_charts import FeatureCharts
from .metrics_charts import MetricsCharts
from .learning_curve_charts import LearningCurveCharts
from .shap_charts import SHAPCharts
from .model_interpretation_charts import ModelInterpretationCharts
from ...common.app_logging.base_app_logger import BaseAppLogger
from ...common.error_handling.base_error_handler import BaseErrorHandler
from ...common.config_management.base_config_manager import BaseConfigManager

class ChartFactory:
    """Factory for creating chart instances with proper dependency injection."""
    
    # Registry of available chart types
    _chart_map: Dict[ChartType, Type[BaseChart]] = {
        ChartType.FEATURE: FeatureCharts,
        ChartType.METRICS: MetricsCharts,
        ChartType.LEARNING_CURVE: LearningCurveCharts,
        ChartType.SHAP: SHAPCharts,
        ChartType.MODEL_INTERPRETATION: ModelInterpretationCharts,
    }

    @classmethod
    def create_chart(cls, 
                    chart_type: ChartType,
                    config: BaseConfigManager,
                    app_logger: BaseAppLogger,
                    error_handler: BaseErrorHandler) -> BaseChart:
        """
        Create a chart instance with dependencies.
        
        Args:
            chart_type: Type of chart to create
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler
            
        Returns:
            Configured chart instance
            
        Raises:
            ValueError: If chart_type is unknown
        """
        try:
            chart_class = cls._chart_map.get(chart_type)
            if chart_class is None:
                raise ValueError(f"Unknown chart type: {chart_type}")
                
            return chart_class(
                config=config,
                app_logger=app_logger,
                error_handler=error_handler
            )
            
        except Exception as e:
            error_handler.create_error_handler(
                'chart_creation',
                f"Error creating chart of type {chart_type}",
                original_error=str(e)
            )

    @classmethod
    def register_chart(cls, 
                      chart_type: ChartType, 
                      chart_class: Type[BaseChart]) -> None:
        """
        Register a new chart type.
        
        Args:
            chart_type: Enum value for the chart type
            chart_class: Chart class to register
        """
        if not issubclass(chart_class, BaseChart):
            raise ValueError(f"Chart class must inherit from BaseChart: {chart_class}")
            
        cls._chart_map[chart_type] = chart_class 