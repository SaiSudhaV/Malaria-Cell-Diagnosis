"""MLflow model tracking and registration"""

import logging
import json
import mlflow
import mlflow.keras
from pathlib import Path
from typing import Dict, Any
from src.utils.config import Config
from src.models.predictor import MalariaPredictor
import numpy as np

logger = logging.getLogger(__name__)


class MLflowManager:
    """Manage MLflow experiments and model registration"""
    
    def __init__(self, tracking_uri: str = None, experiment_name: str = None):
        """Initialize MLflow manager
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of experiment
        """
        self.tracking_uri = tracking_uri or Config.MLFLOW_TRACKING_URI
        self.experiment_name = experiment_name or Config.MLFLOW_EXPERIMENT_NAME
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set experiment
        mlflow.set_experiment(self.experiment_name)
        
        logger.info(f"MLflow configured: {self.tracking_uri} | Experiment: {self.experiment_name}")
    
    def start_run(self, run_name: str = None) -> str:
        """Start a new MLflow run
        
        Args:
            run_name: Name for the run
            
        Returns:
            Run ID
        """
        run = mlflow.start_run(run_name=run_name)
        logger.info(f"MLflow run started: {run.info.run_id}")
        return run.info.run_id
    
    def end_run(self):
        """End current MLflow run"""
        mlflow.end_run()
        logger.info("MLflow run ended")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow
        
        Args:
            params: Dictionary of parameters
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
        logger.info(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to MLflow
        
        Args:
            metrics: Dictionary of metrics
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        logger.info(f"Logged {len(metrics)} metrics")
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log artifact to MLflow
        
        Args:
            local_path: Local file path
            artifact_path: Artifact path in MLflow
        """
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"Logged artifact: {local_path}")
    
    def log_model(self, model, model_name: str = None):
        """Log Keras model to MLflow
        
        Args:
            model: Keras model
            model_name: Name for the model
        """
        model_name = model_name or Config.MODEL_NAME
        mlflow.keras.log_model(model, model_name, registered_model_name=model_name)
        logger.info(f"Model logged: {model_name}")
    
    def register_model(self, model_path: str, model_name: str = None, 
                       model_version_description: str = None):
        """Register trained H5 model in MLflow
        
        Args:
            model_path: Path to H5 model file
            model_name: Name for registered model
            model_version_description: Description for model version
        """
        model_name = model_name or Config.MODEL_NAME
        
        try:
            run_id = self.start_run(f"register_{model_name}")
            
            # Log model parameters
            params = {
                "model_type": "VGG16 Transfer Learning",
                "img_size": str(Config.IMG_SIZE),
                "batch_size": Config.BATCH_SIZE,
                "classes": ",".join(Config.CLASSES),
            }
            self.log_params(params)
            
            # Load model to get info
            predictor = MalariaPredictor(model_path)
            model_info = predictor.get_model_info()
            
            # Log model info as artifacts
            info_file = Path("/tmp/model_info.json")
            with open(info_file, "w") as f:
                json.dump(model_info, f, indent=2)
            self.log_artifact(str(info_file), "model_info")
            
            # Log configuration
            config_dict = {
                "model_name": Config.MODEL_NAME,
                "model_version": Config.MODEL_VERSION,
                "threshold": Config.PREDICTION_THRESHOLD,
                "classes": Config.CLASSES,
            }
            config_file = Path("/tmp/config.json")
            with open(config_file, "w") as f:
                json.dump(config_dict, f, indent=2)
            self.log_artifact(str(config_file), "config")
            
            # Register model
            mlflow.register_model(
                f"runs:/{run_id}/{model_name}",
                model_name
            )
            
            logger.info(f"Model registered successfully: {model_name}")
            self.end_run()
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            self.end_run()
            raise
    
    def log_evaluation_metrics(self, metrics: Dict[str, float], 
                               predictions: np.ndarray = None,
                               labels: np.ndarray = None):
        """Log evaluation metrics to MLflow
        
        Args:
            metrics: Dictionary of metrics (accuracy, precision, recall, etc.)
            predictions: Model predictions
            labels: True labels
        """
        self.log_metrics(metrics)
        
        # Log confusion matrix if available
        if predictions is not None and labels is not None:
            try:
                from sklearn.metrics import confusion_matrix
                import matplotlib.pyplot as plt
                
                cm = confusion_matrix(labels, predictions > 0.5)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)
                ax.set(xticks=np.arange(cm.shape[1]),
                       yticks=np.arange(cm.shape[0]),
                       ylabel='True label',
                       xlabel='Predicted label')
                
                fig.savefig('/tmp/confusion_matrix.png')
                plt.close(fig)
                
                self.log_artifact('/tmp/confusion_matrix.png', 'plots')
                logger.info("Confusion matrix logged")
                
            except Exception as e:
                logger.warning(f"Could not log confusion matrix: {str(e)}")
    
    def get_run_info(self) -> Dict:
        """Get current run information
        
        Returns:
            Dictionary with run info
        """
        run = mlflow.active_run()
        if run:
            return {
                'run_id': run.info.run_id,
                'experiment_id': run.info.experiment_id,
                'status': run.info.status,
            }
        return {}


def register_model_main():
    """Main function to register model"""
    try:
        manager = MLflowManager()
        
        # Start MLflow UI: mlflow ui --host 0.0.0.0 --port 5000
        logger.info("Registering model...")
        
        model_path = str(Config.MODEL_H5)
        manager.register_model(
            model_path,
            model_name=Config.MODEL_NAME,
            model_version_description=f"{Config.MODEL_NAME} v{Config.MODEL_VERSION}"
        )
        
        logger.info("Model registration completed!")
        
    except Exception as e:
        logger.error(f"Failed to register model: {str(e)}")
        raise


if __name__ == "__main__":
    register_model_main()
