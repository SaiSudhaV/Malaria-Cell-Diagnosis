"""Logging configuration and utilities"""

import logging
import sys
from pathlib import Path
from src.utils.config import Config


def setup_logging(log_name: str = "malaria_diagnosis", level=logging.INFO):
    """Setup logging configuration"""
    
    Config.ensure_dirs()
    
    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = Config.LOG_DIR / f"{log_name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str):
    """Get logger instance"""
    return logging.getLogger(name)
