"""Model evaluation script"""

import logging
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loader import ImageDataLoader
from src.models.predictor import MalariaPredictor
from src.utils.config import Config
from src.utils.logging_config import setup_logging
from scripts.register_model_mlflow import MLflowManager

logger = setup_logging("evaluate")


def evaluate_model(data_dir: str = None, split: str = "test", 
                   use_mlflow: bool = False) -> dict:
    """Evaluate model on a dataset
    
    Args:
        data_dir: Data directory path
        split: Data split (train, val, test)
        use_mlflow: Whether to log to MLflow
        
    Returns:
        Dictionary with evaluation results
    """
    if data_dir is None:
        data_dir = str(Config.RAW_DATA_DIR)
    
    logger.info(f"Starting evaluation on {split} split")
    
    # Load data
    loader = ImageDataLoader(data_dir)
    image_paths, labels = loader.load_split_data(split)
    
    logger.info(f"Loaded {len(image_paths)} images")
    
    # Load and preprocess images
    images = loader.load_batch(image_paths)
    
    from src.data.preprocessor import ImagePreprocessor
    preprocessor = ImagePreprocessor()
    images = preprocessor.preprocess_batch(images)
    
    # Load model and predict
    logger.info("Loading model...")
    predictor = MalariaPredictor()
    
    logger.info("Making predictions...")
    predictions = predictor.model.predict(images, verbose=1).flatten()
    
    # Calculate metrics
    pred_labels = (predictions > Config.PREDICTION_THRESHOLD).astype(int)
    
    metrics = {
        'split': split,
        'total_samples': len(labels),
        'timestamp': datetime.now().isoformat(),
        'accuracy': float(accuracy_score(labels, pred_labels)),
        'precision': float(precision_score(labels, pred_labels, zero_division=0)),
        'recall': float(recall_score(labels, pred_labels, zero_division=0)),
        'f1_score': float(f1_score(labels, pred_labels, zero_division=0)),
        'roc_auc': float(roc_auc_score(labels, predictions)),
    }
    
    # Confusion matrix
    cm = confusion_matrix(labels, pred_labels)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report
    class_report = classification_report(
        labels, pred_labels, 
        target_names=Config.CLASSES,
        output_dict=True
    )
    metrics['classification_report'] = class_report
    
    logger.info(f"Evaluation metrics: {metrics}")
    
    # Generate plots
    _generate_evaluation_plots(labels, predictions, pred_labels, cm, split)
    
    # Log to MLflow if enabled
    if use_mlflow:
        try:
            mlflow_manager = MLflowManager()
            mlflow_manager.start_run(f"evaluation_{split}")
            mlflow_manager.log_metrics(metrics)
            mlflow_manager.log_evaluation_metrics(metrics, predictions, labels)
            mlflow_manager.end_run()
            logger.info("Metrics logged to MLflow")
        except Exception as e:
            logger.warning(f"Could not log to MLflow: {str(e)}")
    
    return metrics


def _generate_evaluation_plots(labels, predictions, pred_labels, cm, split):
    """Generate evaluation plots
    
    Args:
        labels: True labels
        predictions: Model predictions (probabilities)
        pred_labels: Binary predictions
        cm: Confusion matrix
        split: Data split name
    """
    output_dir = Config.PROJECT_ROOT / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=Config.CLASSES, yticklabels=Config.CLASSES)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'Confusion Matrix - {split.upper()}')
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_{split}.png", dpi=100)
    plt.close()
    logger.info(f"Saved confusion matrix plot")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, predictions)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', label=f'ROC Curve')
    ax.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {split.upper()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"roc_curve_{split}.png", dpi=100)
    plt.close()
    logger.info(f"Saved ROC curve plot")
    
    # Prediction Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(predictions[labels == 0], bins=30, alpha=0.5, label='Uninfected (True)')
    ax.hist(predictions[labels == 1], bins=30, alpha=0.5, label='Infected (True)')
    ax.axvline(Config.PREDICTION_THRESHOLD, color='r', linestyle='--', 
               label=f'Threshold ({Config.PREDICTION_THRESHOLD})')
    ax.set_xlabel('Prediction Score')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Prediction Score Distribution - {split.upper()}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"prediction_distribution_{split}.png", dpi=100)
    plt.close()
    logger.info(f"Saved prediction distribution plot")


def save_evaluation_results(metrics: dict, output_path: str = None):
    """Save evaluation results to JSON
    
    Args:
        metrics: Evaluation metrics dictionary
        output_path: Output file path
    """
    if output_path is None:
        output_dir = Config.PROJECT_ROOT / "evaluation_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"evaluation_{metrics['split']}.json"
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved evaluation results to {output_path}")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate malaria diagnosis model')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Data directory path')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Data split to evaluate')
    parser.add_argument('--mlflow', action='store_true',
                        help='Log results to MLflow')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path for results')
    
    args = parser.parse_args()
    
    try:
        # Run evaluation
        metrics = evaluate_model(
            data_dir=args.data_dir,
            split=args.split,
            use_mlflow=args.mlflow
        )
        
        # Save results
        save_evaluation_results(metrics, args.output)
        
        # Print summary
        print("\n" + "="*60)
        print(f"EVALUATION SUMMARY - {args.split.upper()} SET")
        print("="*60)
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"Accuracy:      {metrics['accuracy']:.4f}")
        print(f"Precision:     {metrics['precision']:.4f}")
        print(f"Recall:        {metrics['recall']:.4f}")
        print(f"F1 Score:      {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:       {metrics['roc_auc']:.4f}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
