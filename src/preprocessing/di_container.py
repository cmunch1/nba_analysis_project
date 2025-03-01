from dependency_injector import containers, providers

from src.common.common_di_container import CommonDIContainer
from .preprocessor_factory import PreprocessorFactory


class PreprocessingDIContainer(containers.DeclarativeContainer):
    """Container for preprocessing dependencies."""
    
    # Import common container
    common = providers.Container(CommonDIContainer)
    
    # Use common container's components
    config = common.config
    app_logger = common.app_logger  # Changed from common.logger
    app_file_handler = common.app_file_handler
    error_handler = common.error_handler_factory
    
    # Create preprocessor factory
    preprocessor = providers.Factory(
        PreprocessorFactory.create_preprocessor,
        config=config,
        app_logger=app_logger,
        app_file_handler=app_file_handler,
        error_handler=error_handler,
        preprocessor_type="modular"
    )
    
    @classmethod
    def configure_preprocessor(cls, preprocessor_type: str) -> None:
        """
        Configure the container to use a different preprocessor type.
        
        Args:
            preprocessor_type: Type of preprocessor to use
        """
        cls.preprocessor.override(
            providers.Factory(
                PreprocessorFactory.create_preprocessor,
                config=cls.config,
                app_logger=cls.app_logger,
                app_file_handler=cls.app_file_handler,
                error_handler=cls.error_handler,
                preprocessor_type=preprocessor_type
            )
        )