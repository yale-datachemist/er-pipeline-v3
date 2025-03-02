import logging
import atexit
from datetime import datetime
import os
import sys

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
    
    # Get the root logger and configure it
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Close and remove any existing handlers to avoid ResourceWarnings
    for handler in list(root_logger.handlers):
        # Properly close the handler
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass  # Ignore errors during closure
        
        # Remove it from the logger
        root_logger.removeHandler(handler)
    
    # Create fresh handlers
    file_handler = logging.FileHandler(log_file, 'w')  # 'w' mode for clean start
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Configure format for both handlers
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Store handlers for later cleanup
    global _log_handlers
    _log_handlers = [file_handler, console_handler]
    
    # Register function to close handlers on exit
    def close_handlers():
        for handler in _log_handlers:
            try:
                handler.flush()
                handler.close()
                root_logger.removeHandler(handler)
            except Exception:
                pass  # Ignore errors during cleanup
    
    # Make sure we register the cleanup only once
    if not hasattr(setup_logging, "_registered"):
        atexit.register(close_handlers)
        setup_logging._registered = True
    
    # Create a logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

# Global variable to store handlers for cleanup
_log_handlers = []

# Ensure cleanup happens on module unload
def _cleanup_on_unload():
    for handler in _log_handlers:
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass

# Register module unload cleanup
atexit.register(_cleanup_on_unload)