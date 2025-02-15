from typing import Type
from src.common.config_management.base_config_manager import BaseConfigManager
from src.common.app_logging.base_app_logger import BaseAppLogger
from src.common.data_access.base_data_access import BaseDataAccess
from src.common.data_access.csv_data_access import CSVDataAccess
from src.common.file_handling.base_file_handler import BaseFileHandler

class DataAccessFactory:
    def __init__(self, data_access_class: Type[BaseDataAccess] = CSVDataAccess):
        self.data_access_class = data_access_class
        
    def create_data_access(self, config: BaseConfigManager, logger: BaseAppLogger, 
                          file_handler: BaseFileHandler) -> BaseDataAccess:
        return self.data_access_class(config, logger, file_handler)