"""Batch prediction script"""

import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from src.data.loader import ImageDataLoader
from src.models.predictor import MalariaPredictor
from src.utils.config import Config
from src.utils.logging_config import setup_logging
from workflows.monitoring import ModelMonitor

logger = setup_logging("batch_predict")


def batch_predict(image_paths: List[str], save_results: bool = True,
                 output_dir: str = None) -> List[Dict]:
    """Make predictions on batch of images
    
    Args:
        image_paths: List of image file paths
        save_results: Whether to save results to file
        output_dir: Output directory for results
        
    Returns:
        List of prediction results
    """
    logger.info(f"Starting batch prediction on {len(image_paths)} images")
    
    # Load images
    loader = ImageDataLoader()
    images = loader.load_batch(image_paths)
    
    # Load model
    logger.info("Loading model...")
    predictor = MalariaPredictor()
    
    # Make predictions
    logger.info("Making predictions...")
    results = predictor.predict_batch(images)
    
    # Add file paths to results
    for i, result in enumerate(results):
        if i < len(image_paths):
            result['image_path'] = image_paths[i]
    
    # Save results
    if save_results:
        if output_dir is None:
            output_dir = Config.PROJECT_ROOT / "predictions"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"batch_predictions_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        # Also save summary
        summary = {
            'total_images': len(results),
            'timestamp': datetime.now().isoformat(),
            'infected_count': sum(1 for r in results if r['class_name'] == 'Infected'),
            'uninfected_count': sum(1 for r in results if r['class_name'] == 'Uninfected'),
            'avg_confidence': sum(r['confidence'] for r in results) / len(results) if results else 0,
        }
        
        summary_file = output_dir / f"batch_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to {summary_file}")
    
    return results


def predict_from_directory(directory: str, save_results: bool = True,
                          output_dir: str = None) -> List[Dict]:
    """Make predictions on all images in a directory
    
    Args:
        directory: Directory containing images
        save_results: Whether to save results
        output_dir: Output directory for results
        
    Returns:
        List of prediction results
    """
    logger.info(f"Loading images from {directory}")
    loader = ImageDataLoader()
    image_paths = loader.list_images_in_directory(directory)
    
    if not image_paths:
        logger.warning(f"No images found in {directory}")
        return []
    
    return batch_predict(image_paths, save_results, output_dir)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Batch prediction for malaria diagnosis')
    parser.add_argument('--directory', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results')
    parser.add_argument('--monitor', action='store_true',
                        help='Log predictions for monitoring')
    
    args = parser.parse_args()
    
    try:
        # Run predictions
        results = predict_from_directory(
            args.directory,
            save_results=not args.no_save,
            output_dir=args.output
        )
        
        # Log for monitoring
        if args.monitor:
            monitor = ModelMonitor()
            for result in results:
                monitor.log_prediction(result)
            logger.info("Predictions logged for monitoring")
        
        # Print summary
        print("\n" + "="*60)
        print("BATCH PREDICTION SUMMARY")
        print("="*60)
        print(f"Total Images: {len(results)}")
        
        infected = sum(1 for r in results if r['class_name'] == 'Infected')
        uninfected = sum(1 for r in results if r['class_name'] == 'Uninfected')
        
        print(f"Infected: {infected} ({infected/len(results)*100:.1f}%)")
        print(f"Uninfected: {uninfected} ({uninfected/len(results)*100:.1f}%)")
        
        if results:
            avg_conf = sum(r['confidence'] for r in results) / len(results)
            print(f"Avg Confidence: {avg_conf:.4f}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
