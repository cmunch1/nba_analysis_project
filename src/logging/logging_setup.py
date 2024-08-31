import os
import logging
from datetime import datetime, timezone

from logging.handlers import RotatingFileHandler
import uuid
from pythonjsonlogger import jsonlogger

from ..config.abstract_config import AbstractConfig

class ErrorIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'error_id'):
            record.error_id = str(uuid.uuid4())
        return True

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get('timestamp'):
            # Convert to ISO 8601 format
            now = datetime.now(timezone.utc).isoformat()
            log_record['timestamp'] = now
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname

def setup_logging(config: AbstractConfig, filename: str):
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    error_logger = logging.getLogger('error_logger')
    error_logger.setLevel(logging.ERROR)
    error_logger.addFilter(ErrorIdFilter())

    json_formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(name)s %(message)s')

    if config.log_to_file:
        log_filename = filename if filename else config.log_filename_fallback
        full_log_path = os.path.join(config.log_path, log_filename)
        
        os.makedirs(os.path.dirname(full_log_path), exist_ok=True)
        
        file_mode = 'w' if config.log_current_run_only else 'a'
        
        file_handler = RotatingFileHandler(
            full_log_path,
            mode=file_mode,
            maxBytes=config.log_size_max*1024*1024,  
            backupCount=config.log_backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, config.log_level))
        file_handler.setFormatter(json_formatter)
        logging.getLogger('').addHandler(file_handler)

        error_file_handler = RotatingFileHandler(
            os.path.join(config.log_path, config.error_filename_fallback),
            maxBytes=config.error_size_max*10*1024*1024,  
            backupCount=config.error_backup_count,
            encoding='utf-8'
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(json_formatter)
        error_logger.addHandler(error_file_handler)

    if config.log_to_console:
        console = logging.StreamHandler()
        console.setLevel(getattr(logging, config.log_level))
        console.setFormatter(json_formatter)
        logging.getLogger('').addHandler(console)

    return error_logger
