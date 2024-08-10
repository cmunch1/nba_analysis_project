import os
import logging
from ..config.config import config

def setup_logging(filename=None):
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if config.log_to_file:
        log_filename = filename if filename else "logger_file.log"
        full_log_path = os.path.join(config.log_path, log_filename)
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(full_log_path), exist_ok=True)
        
        # Check if we should only log the current run
        file_mode = 'w' if config.log_current_run_only else 'a'
        
        file_handler = logging.FileHandler(full_log_path, mode=file_mode)
        file_handler.setLevel(getattr(logging, config.log_level))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger('').addHandler(file_handler)

    if config.log_to_console:
        console = logging.StreamHandler()
        console.setLevel(getattr(logging, config.log_level))
        console.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger('').addHandler(console)
