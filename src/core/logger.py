"""
Step 27: Centralized Logging Module.
Handles system-wide logging to file and console with rotation.
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from src.core.config import LOGS_DIR

class SystemLogger:
    """
    Singleton Logger for the Trading System.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SystemLogger, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
        
    def _initialize(self):
        self.logger = logging.getLogger("VisionTrade")
        self.logger.setLevel(logging.INFO)
        
        if not os.path.exists(LOGS_DIR):
            os.makedirs(LOGS_DIR, exist_ok=True)
            
        # File Handler (Rotate after 5MB)
        log_file = os.path.join(LOGS_DIR, "system.log")
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
    def info(self, msg: str):
        self.logger.info(msg)
        
    def error(self, msg: str):
        self.logger.error(msg)
        
    def warning(self, msg: str):
        self.logger.warning(msg)
        
    def debug(self, msg: str):
        self.logger.debug(msg)
