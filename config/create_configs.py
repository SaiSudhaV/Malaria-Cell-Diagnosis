"""Configuration files creation"""

import yaml
import json
from pathlib import Path

def create_config_files():
    """Create all configuration files"""
    
    config_dir = Path(__file__).parent
    
    # config.yaml
    config = {
        'model': {
            'name': 'malaria_vgg16',
            'version': '1.0',
            'type': 'VGG16 Transfer Learning',
            'input_shape': [224, 224, 3],
            'threshold': 0.5,
        },
        'data': {
            'img_size': [224, 224],
            'batch_size': 32,
            'classes': ['Uninfected', 'Infected'],
            'splits': {
                'train': 0.8,
                'val': 0.1,
                'test': 0.1,
            }
        },
        'training': {
            'epochs': 8,
            'learning_rate': 1e-4,
            'optimizer': 'Adam',
            'loss': 'binary_crossentropy',
        },
        'api': {
            'host': '0.0.0.0',
            'port': 8000,
            'debug': False,
        },
        'mlflow': {
            'tracking_uri': 'http://localhost:5000',
            'experiment_name': 'malaria-diagnosis',
        }
    }
    
    with open(config_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created config.yaml")
    
    # logging.yaml
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s - %(funcName)s:%(lineno)d - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': 'logs/app.log',
                'maxBytes': 10485760,
                'backupCount': 5
            }
        },
        'loggers': {
            'src': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'api': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console', 'file']
        }
    }
    
    with open(config_dir / 'logging.yaml', 'w') as f:
        yaml.dump(logging_config, f, default_flow_style=False)
    
    print(f"Created logging.yaml")


if __name__ == "__main__":
    create_config_files()
