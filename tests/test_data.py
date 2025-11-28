"""Unit tests for data loading"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
import os

from src.data.loader import ImageDataLoader
from src.data.preprocessor import ImagePreprocessor


@pytest.fixture
def temp_image_dir():
    """Create temporary directory with test images"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create class directories
        class_dirs = [
            Path(tmpdir) / "Uninfected",
            Path(tmpdir) / "Infected"
        ]
        
        for class_dir in class_dirs:
            class_dir.mkdir()
            
            # Create test images
            for i in range(3):
                img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
                img.save(class_dir / f"image_{i}.jpg")
        
        yield tmpdir


def test_image_loader_load_image(temp_image_dir):
    """Test loading a single image"""
    loader = ImageDataLoader(temp_image_dir)
    
    image_path = Path(temp_image_dir) / "Uninfected" / "image_0.jpg"
    image = loader.load_image(str(image_path))
    
    assert isinstance(image, np.ndarray)
    assert image.shape == (224, 224, 3)


def test_image_loader_load_batch(temp_image_dir):
    """Test loading a batch of images"""
    loader = ImageDataLoader(temp_image_dir)
    
    image_paths = [
        str(Path(temp_image_dir) / "Uninfected" / "image_0.jpg"),
        str(Path(temp_image_dir) / "Uninfected" / "image_1.jpg"),
    ]
    
    images = loader.load_batch(image_paths)
    
    assert isinstance(images, np.ndarray)
    assert images.shape[0] == 2
    assert images.shape == (2, 224, 224, 3)


def test_image_loader_list_images(temp_image_dir):
    """Test listing images in directory"""
    loader = ImageDataLoader(temp_image_dir)
    
    images = loader.list_images_in_directory(temp_image_dir)
    
    assert len(images) == 6  # 3 images per class


def test_preprocessor_preprocess(temp_image_dir):
    """Test image preprocessing"""
    loader = ImageDataLoader(temp_image_dir)
    preprocessor = ImagePreprocessor()
    
    image = loader.load_image(str(Path(temp_image_dir) / "Uninfected" / "image_0.jpg"))
    processed = preprocessor.preprocess(image)
    
    assert isinstance(processed, np.ndarray)
    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.float32
    assert processed.min() >= 0 and processed.max() <= 1


def test_preprocessor_preprocess_batch(temp_image_dir):
    """Test batch preprocessing"""
    loader = ImageDataLoader(temp_image_dir)
    preprocessor = ImagePreprocessor()
    
    image_paths = [
        str(Path(temp_image_dir) / "Uninfected" / "image_0.jpg"),
        str(Path(temp_image_dir) / "Uninfected" / "image_1.jpg"),
    ]
    
    images = loader.load_batch(image_paths)
    processed = preprocessor.preprocess_batch(images)
    
    assert isinstance(processed, np.ndarray)
    assert processed.shape[0] == 2
    assert processed.dtype == np.float32
