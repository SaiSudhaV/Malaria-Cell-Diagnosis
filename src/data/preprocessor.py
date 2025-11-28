"""Data preprocessing utilities"""

import numpy as np
import cv2
from typing import Tuple
from src.utils.config import Config
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocess images for model inference"""
    
    def __init__(self, img_size: Tuple[int, int] = None, normalize: bool = True):
        """Initialize preprocessor
        
        Args:
            img_size: Target image size (height, width)
            normalize: Whether to normalize image values to [0, 1]
        """
        self.img_size = img_size or Config.IMG_SIZE
        self.normalize = normalize
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a single image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Ensure image is in RGB format
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize to target size
        image = cv2.resize(image, self.img_size)
        
        # Normalize to [0, 1]
        if self.normalize:
            image = image.astype(np.float32) / 255.0
        
        return image
    
    def preprocess_batch(self, images: np.ndarray) -> np.ndarray:
        """Preprocess a batch of images
        
        Args:
            images: Batch of images (N, H, W, C)
            
        Returns:
            Preprocessed batch
        """
        processed = []
        
        for image in images:
            processed_image = self.preprocess(image)
            processed.append(processed_image)
        
        return np.array(processed)
    
    def augment(self, image: np.ndarray, 
                rotation_range: int = 20,
                width_shift: float = 0.1,
                height_shift: float = 0.1,
                horizontal_flip: bool = True) -> np.ndarray:
        """Apply data augmentation to image
        
        Args:
            image: Input image
            rotation_range: Rotation range in degrees
            width_shift: Fraction of width to shift
            height_shift: Fraction of height to shift
            horizontal_flip: Whether to apply horizontal flip
            
        Returns:
            Augmented image
        """
        height, width = image.shape[:2]
        
        # Random rotation
        angle = np.random.uniform(-rotation_range, rotation_range)
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        # Random shift
        shift_x = int(np.random.uniform(-width_shift, width_shift) * width)
        shift_y = int(np.random.uniform(-height_shift, height_shift) * height)
        shift_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        image = cv2.warpAffine(image, shift_matrix, (width, height))
        
        # Random horizontal flip
        if horizontal_flip and np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
        
        return image
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def standardize(image: np.ndarray, 
                   mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                   std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
        """Standardize image using ImageNet statistics
        
        Args:
            image: Input image (normalized to [0, 1])
            mean: Channel-wise mean values
            std: Channel-wise standard deviation values
            
        Returns:
            Standardized image
        """
        image = image.copy()
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            for i in range(3):
                image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        
        return image
