from typing import Type
from src.common.config_management.base_config_manager import BaseConfigManager
from src.common.app_logging.base_app_logger import BaseAppLogger
from src.common.data_access.base_data_access import BaseDataAccess
from src.common.data_access.csv_data_access import CSVDataAccess
from src.common.app_file_handling.base_app_file_handler import BaseAppFileHandler
from src.common.error_handling.base_error_handler import BaseErrorHandler

class DataAccessFactory:
    def __init__(self):
        pass
        
    def create_data_access(self, config: BaseConfigManager, app_logger: BaseAppLogger, 
                          app_file_handler: BaseAppFileHandler, error_handler: BaseErrorHandler,
                          data_access_class: Type[BaseDataAccess] = CSVDataAccess) -> BaseDataAccess:
        return data_access_class(config, app_logger, app_file_handler, error_handler)