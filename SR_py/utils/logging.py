# -*- coding: utf-8 -*-
"""
Logging configuration for the sisso_py library.
"""
import logging
import sys

def setup_logging(level=logging.INFO):
    """
    Sets up a basic logger that prints to stdout.
    """
    logger = logging.getLogger('sisso_py')
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(level)
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(handler)
    
    return logger

# Initialize logger with a default configuration
# This can be called from the main entry point of the library or application.
# setup_logging()
