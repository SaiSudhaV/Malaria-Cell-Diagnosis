"""Model prediction and inference utilities"""

import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import tensorflow as tf
from src.utils.config import Config
from src.data.preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)


class MalariaPredictor:
    """Wrapper for malaria diagnosis model inference"""
    
    def __init__(self, model_path: str = None):
        """Initialize predictor with trained model
        
        Args:
            model_path: Path to H5 model file
        """
        self.model_path = Path(model_path) if model_path else Config.MODEL_H5
        self.model = None
        self.preprocessor = ImagePreprocessor()
        self.threshold = Config.PREDICTION_THRESHOLD
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            logger.info(f"Loading model from {self.model_path}")
            
            # Suppress TensorFlow warnings
            tf.get_logger().setLevel('ERROR')
            
            # Load model
            self.model = tf.keras.models.load_model(str(self.model_path))
            
            logger.info("Model loaded successfully")
            self.print_model_info()
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def print_model_info(self):
        """Print model architecture and summary"""
        logger.info(f"Model: {Config.MODEL_NAME} v{Config.MODEL_VERSION}")
        logger.info(f"Input shape: {self.model.input_shape}")
        logger.info(f"Output shape: {self.model.output_shape}")
    
    def predict(self, image: np.ndarray, return_confidence: bool = True) -> Dict:
        """Predict on a single image
        
        Args:
            image: Input image as numpy array
            return_confidence: Whether to return confidence score
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess image
            processed_image = self.preprocessor.preprocess(image)
            
            # Add batch dimension
            image_batch = np.expand_dims(processed_image, axis=0)
            
            # Predict
            prediction = self.model.predict(image_batch, verbose=0)[0][0]
            
            # Determine class
            class_idx = int(prediction > self.threshold)
            class_name = Config.CLASS_INDICES[class_idx]
            
            # Calculate confidence
            if prediction > 0.5:
                confidence = float(prediction)
            else:
                confidence = float(1 - prediction)
            
            result = {
                'class_name': class_name,
                'class_id': class_idx,
                'raw_prediction': float(prediction),
                'confidence': confidence,
                'threshold': self.threshold
            }
            
            logger.debug(f"Prediction: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def predict_batch(self, images: np.ndarray) -> List[Dict]:
        """Predict on a batch of images
        
        Args:
            images: Batch of images (N, H, W, C)
            
        Returns:
            List of prediction dictionaries
        """
        try:
            # Preprocess batch
            processed_images = self.preprocessor.preprocess_batch(images)
            
            # Predict
            predictions = self.model.predict(processed_images, verbose=0)
            
            results = []
            for i, pred in enumerate(predictions):
                pred_value = float(pred[0])
                class_idx = int(pred_value > self.threshold)
                class_name = Config.CLASS_INDICES[class_idx]
                
                confidence = pred_value if pred_value > 0.5 else 1 - pred_value
                
                result = {
                    'image_index': i,
                    'class_name': class_name,
                    'class_id': class_idx,
                    'raw_prediction': pred_value,
                    'confidence': float(confidence),
                    'threshold': self.threshold
                }
                results.append(result)
            
            logger.info(f"Batch prediction completed for {len(images)} images")
            return results
            
        except Exception as e:
            logger.error(f"Error during batch prediction: {str(e)}")
            raise
    
    def predict_from_file(self, image_path: str) -> Dict:
        """Predict on image from file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Prediction results
        """
        try:
            import cv2
            
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            result = self.predict(image)
            result['image_path'] = image_path
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting from file {image_path}: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get model information
        
        Returns:
            Dictionary with model details
        """
        return {
            'model_name': Config.MODEL_NAME,
            'version': Config.MODEL_VERSION,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'classes': Config.CLASSES,
            'threshold': self.threshold,
            'model_path': str(self.model_path)
        }
