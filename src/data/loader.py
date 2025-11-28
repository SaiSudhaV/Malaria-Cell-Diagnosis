"""Data loading utilities"""

import os
import logging
from pathlib import Path
from typing import Tuple, List
import numpy as np
import cv2
from PIL import Image
from src.utils.config import Config

logger = logging.getLogger(__name__)


class ImageDataLoader:
    """Load and manage image data"""
    
    def __init__(self, data_dir: str = None):
        """Initialize data loader
        
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = Path(data_dir) if data_dir else Config.RAW_DATA_DIR
        self.img_size = Config.IMG_SIZE
        
    def load_image(self, image_path: str) -> np.ndarray:
        """Load a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Load image using OpenCV
            image = cv2.imread(image_path)
            
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            image = cv2.resize(image, self.img_size)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise
    
    def load_batch(self, image_paths: List[str]) -> np.ndarray:
        """Load a batch of images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Batch of images as numpy array (N, H, W, C)
        """
        images = []
        for path in image_paths:
            try:
                image = self.load_image(path)
                images.append(image)
            except Exception as e:
                logger.warning(f"Skipping image {path}: {str(e)}")
                continue
        
        if not images:
            raise ValueError("No valid images loaded")
        
        return np.array(images)
    
    def list_images_in_directory(self, directory: str) -> List[str]:
        """List all images in a directory
        
        Args:
            directory: Directory path
            
        Returns:
            List of image file paths
        """
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in valid_extensions:
                    image_paths.append(os.path.join(root, file))
        
        logger.info(f"Found {len(image_paths)} images in {directory}")
        return image_paths
    
    def load_class_data(self, class_name: str) -> List[str]:
        """Load all images from a specific class directory
        
        Args:
            class_name: Name of class directory
            
        Returns:
            List of image file paths
        """
        class_dir = self.data_dir / class_name
        
        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        
        return self.list_images_in_directory(str(class_dir))
    
    def load_split_data(self, split: str = "train") -> Tuple[List[str], np.ndarray]:
        """Load data for a specific split (train/val/test)
        
        Args:
            split: Data split name (train, val, test)
            
        Returns:
            Tuple of (image_paths, labels)
        """
        split_dir = self.data_dir / split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        image_paths = []
        labels = []
        
        for class_idx, class_name in enumerate(Config.CLASSES):
            class_dir = split_dir / class_name
            
            if class_dir.exists():
                class_images = self.list_images_in_directory(str(class_dir))
                image_paths.extend(class_images)
                labels.extend([class_idx] * len(class_images))
        
        logger.info(f"Loaded {len(image_paths)} images from {split} split")
        return image_paths, np.array(labels)
    
    def validate_images(self, directory: str) -> Tuple[int, int]:
        """Validate images in directory
        
        Args:
            directory: Directory path
            
        Returns:
            Tuple of (total_images, valid_images)
        """
        image_paths = self.list_images_in_directory(directory)
        total = len(image_paths)
        valid = 0
        
        for path in image_paths:
            try:
                img = cv2.imread(path)
                if img is not None:
                    valid += 1
                else:
                    logger.warning(f"Invalid image: {path}")
            except Exception as e:
                logger.warning(f"Error validating {path}: {str(e)}")
        
        logger.info(f"Validation: {valid}/{total} images valid")
        return total, valid
