import logging
import atexit
from datetime import datetime
import os

# Create a function to configure logging
def setup_logging(log_dir=None):
    """
    Set up logging with proper file handler management.
    
    Args:
        log_dir: Optional directory for log files
    
    Returns:
        The configured logger
    """
    # Create timestamp for log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Determine log file path
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"entity_resolution_{timestamp}.log")
    else:
        log_file = f"entity_resolution_{timestamp}.log"
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Configure format for both handlers
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Get the root logger and configure it
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Add the handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Register function to close handlers on exit
    def close_handlers():
        for handler in root_logger.handlers:
            handler.close()
            root_logger.removeHandler(handler)
            
    atexit.register(close_handlers)
    
    # Create a logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger