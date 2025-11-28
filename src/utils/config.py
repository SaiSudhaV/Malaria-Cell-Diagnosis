"""Configuration module for the project"""

import os
from pathlib import Path
from typing import Optional
import yaml

class Config:
    """Project configuration class"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    TEST_SAMPLES_DIR = DATA_DIR / "test_samples"
    
    MODELS_DIR = PROJECT_ROOT / "models"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
    SCRIPTS_DIR = PROJECT_ROOT / "scripts"
    API_DIR = PROJECT_ROOT / "api"
    WORKFLOWS_DIR = PROJECT_ROOT / "workflows"
    CONFIG_DIR = PROJECT_ROOT / "config"
    
    # Model paths
    MODEL_H5 = MODELS_DIR / "malaria_vgg16_final.h5"
    SAVED_MODEL_DIR = MODELS_DIR / "saved_model"
    
    # MLflow configuration
    MLFLOW_TRACKING_URI = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME = "malaria-diagnosis"
    MLFLOW_ARTIFACTS_DIR = PROJECT_ROOT / "mlruns"
    
    # DVC configuration
    DVC_REMOTE_DIR = DATA_DIR / ".dvc_storage"
    
    # Image configuration
    IMG_SIZE = (224, 224)
    IMG_CHANNELS = 3
    BATCH_SIZE = 32
    
    # Model configuration
    MODEL_NAME = "malaria_vgg16"
    MODEL_VERSION = "1.0"
    
    # Classes
    CLASSES = ["Infected", "Uninfected"]
    CLASS_INDICES = {0: "Uninfected", 1: "Infected"}
    
    # Data splits
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Inference configuration
    PREDICTION_THRESHOLD = 0.5
    CONFIDENCE_THRESHOLD = 0.7
    
    # Monitoring configuration
    EVIDENTLY_REPORTS_DIR = PROJECT_ROOT / "monitoring" / "reports"
    EVIDENTLY_DASHBOARDS_DIR = PROJECT_ROOT / "monitoring" / "dashboards"
    
    # API configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_DEBUG = False
    
    # Logging configuration
    LOG_LEVEL = "INFO"
    LOG_DIR = PROJECT_ROOT / "logs"
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure all required directories exist"""
        dirs = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.TEST_SAMPLES_DIR,
            cls.MODELS_DIR,
            cls.LOG_DIR,
            cls.MLFLOW_ARTIFACTS_DIR,
            cls.EVIDENTLY_REPORTS_DIR,
            cls.EVIDENTLY_DASHBOARDS_DIR,
            cls.DVC_REMOTE_DIR,
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def load_from_yaml(cls, config_path: Optional[str] = None):
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = cls.CONFIG_DIR / "config.yaml"
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                for key, value in config_data.items():
                    if hasattr(cls, key):
                        setattr(cls, key, value)


# Create directories on import
Config.ensure_dirs()
