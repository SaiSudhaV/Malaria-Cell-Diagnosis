"""Prefect workflow orchestration"""

import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
from prefect import flow, task, get_run_context
from prefect.task_runs import task_run_ctx

from src.data.loader import ImageDataLoader
from src.data.preprocessor import ImagePreprocessor
from src.models.predictor import MalariaPredictor
from src.utils.config import Config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)


@task(name="Load Images", retries=2)
def load_images(data_dir: str, split: str = "test") -> tuple:
    """Load images from a data split
    
    Args:
        data_dir: Data directory path
        split: Data split (train, val, test)
        
    Returns:
        Tuple of (image_paths, labels)
    """
    logger.info(f"Loading {split} images from {data_dir}")
    loader = ImageDataLoader(data_dir)
    image_paths, labels = loader.load_split_data(split)
    logger.info(f"Loaded {len(image_paths)} images")
    return image_paths, labels


@task(name="Preprocess Images", retries=2)
def preprocess_images(image_paths: List[str]) -> np.ndarray:
    """Preprocess a batch of images
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        Preprocessed images array
    """
    logger.info(f"Preprocessing {len(image_paths)} images")
    loader = ImageDataLoader()
    preprocessor = ImagePreprocessor()
    
    images = loader.load_batch(image_paths)
    processed = preprocessor.preprocess_batch(images)
    logger.info("Preprocessing completed")
    return processed


@task(name="Model Evaluation", retries=2)
def evaluate_model(image_paths: List[str], labels: np.ndarray) -> Dict:
    """Evaluate model on test data
    
    Args:
        image_paths: List of image file paths
        labels: True labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Starting model evaluation")
    
    # Load and preprocess images
    loader = ImageDataLoader()
    images = loader.load_batch(image_paths)
    
    preprocessor = ImagePreprocessor()
    images = preprocessor.preprocess_batch(images)
    
    # Make predictions
    predictor = MalariaPredictor()
    predictions = predictor.model.predict(images, verbose=0).flatten()
    
    # Calculate metrics
    pred_labels = (predictions > Config.PREDICTION_THRESHOLD).astype(int)
    
    metrics = {
        'accuracy': float(accuracy_score(labels, pred_labels)),
        'precision': float(precision_score(labels, pred_labels, zero_division=0)),
        'recall': float(recall_score(labels, pred_labels, zero_division=0)),
        'f1_score': float(f1_score(labels, pred_labels, zero_division=0)),
    }
    
    # Confusion matrix
    cm = confusion_matrix(labels, pred_labels)
    metrics['confusion_matrix'] = cm.tolist()
    
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


@task(name="Batch Prediction", retries=2)
def batch_predict(image_paths: List[str]) -> List[Dict]:
    """Predict on a batch of images
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        List of prediction results
    """
    logger.info(f"Making predictions on {len(image_paths)} images")
    
    loader = ImageDataLoader()
    images = loader.load_batch(image_paths)
    
    predictor = MalariaPredictor()
    results = predictor.predict_batch(images)
    
    logger.info(f"Predictions completed")
    return results


@task(name="Save Results")
def save_results(results: List[Dict], output_dir: str = None):
    """Save results to file
    
    Args:
        results: List of results
        output_dir: Output directory
    """
    import json
    
    if output_dir is None:
        output_dir = str(Config.PROJECT_ROOT / "predictions")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_file = Path(output_dir) / "batch_predictions.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


# Workflow definitions

@flow(name="Data Preprocessing Flow")
def preprocessing_flow(data_dir: str = None, split: str = "test") -> np.ndarray:
    """Flow for data loading and preprocessing
    
    Args:
        data_dir: Data directory path
        split: Data split to process
        
    Returns:
        Preprocessed images
    """
    if data_dir is None:
        data_dir = str(Config.RAW_DATA_DIR)
    
    image_paths, _ = load_images(data_dir, split)
    processed_images = preprocess_images(image_paths)
    return processed_images


@flow(name="Model Evaluation Flow")
def evaluation_flow(data_dir: str = None, split: str = "test") -> Dict:
    """Flow for model evaluation
    
    Args:
        data_dir: Data directory path
        split: Data split for evaluation
        
    Returns:
        Evaluation metrics
    """
    if data_dir is None:
        data_dir = str(Config.RAW_DATA_DIR)
    
    image_paths, labels = load_images(data_dir, split)
    metrics = evaluate_model(image_paths, labels)
    return metrics


@flow(name="Batch Prediction Flow")
def batch_prediction_flow(data_dir: str = None, split: str = "test", 
                         save_predictions: bool = True) -> List[Dict]:
    """Flow for batch predictions
    
    Args:
        data_dir: Data directory path
        split: Data split for predictions
        save_predictions: Whether to save results
        
    Returns:
        List of prediction results
    """
    if data_dir is None:
        data_dir = str(Config.RAW_DATA_DIR)
    
    image_paths, _ = load_images(data_dir, split)
    results = batch_predict(image_paths)
    
    if save_predictions:
        save_results(results)
    
    return results


@flow(name="End-to-End Pipeline")
def end_to_end_pipeline(data_dir: str = None) -> Dict:
    """Complete end-to-end pipeline
    
    Args:
        data_dir: Data directory path
        
    Returns:
        Dictionary with all results
    """
    if data_dir is None:
        data_dir = str(Config.RAW_DATA_DIR)
    
    logger.info("Starting end-to-end pipeline")
    
    # Preprocessing
    logger.info("Step 1: Data Preprocessing")
    preprocessing_flow(data_dir, "train")
    
    # Evaluation on test set
    logger.info("Step 2: Model Evaluation")
    eval_metrics = evaluation_flow(data_dir, "test")
    
    # Batch predictions
    logger.info("Step 3: Batch Predictions")
    predictions = batch_prediction_flow(data_dir, "test")
    
    results = {
        'evaluation_metrics': eval_metrics,
        'total_predictions': len(predictions),
        'sample_predictions': predictions[:5] if predictions else []
    }
    
    logger.info("End-to-end pipeline completed")
    return results


if __name__ == "__main__":
    # Run flows
    logger.info("Starting Prefect workflows")
    
    # Run end-to-end pipeline
    result = end_to_end_pipeline()
    print("\n" + "="*50)
    print("Pipeline Results:")
    print("="*50)
    print(f"Evaluation Metrics: {result['evaluation_metrics']}")
    print(f"Total Predictions: {result['total_predictions']}")
