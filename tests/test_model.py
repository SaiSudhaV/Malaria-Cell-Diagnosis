"""Unit tests for model inference"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.models.predictor import MalariaPredictor


@pytest.fixture
def mock_model():
    """Create a mock model for testing"""
    mock = Mock()
    mock.predict = Mock(return_value=np.array([[0.8]]))
    mock.input_shape = (None, 224, 224, 3)
    mock.output_shape = (None, 1)
    return mock


def test_predictor_initialization():
    """Test predictor initialization (without actual model)"""
    with patch('src.models.predictor.tf.keras.models.load_model'):
        predictor = MalariaPredictor(model_path="/fake/path/model.h5")
        
        # Check attributes
        assert predictor.model_path == Path("/fake/path/model.h5")
        assert predictor.threshold == 0.5


def test_predictor_get_model_info():
    """Test getting model information"""
    with patch('src.models.predictor.tf.keras.models.load_model') as mock_load:
        mock_model = Mock()
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.output_shape = (None, 1)
        mock_load.return_value = mock_model
        
        predictor = MalariaPredictor(model_path="/fake/path/model.h5")
        info = predictor.get_model_info()
        
        assert 'model_name' in info
        assert 'version' in info
        assert 'classes' in info
        assert len(info['classes']) == 2


def test_predictor_predict():
    """Test prediction on single image"""
    with patch('src.models.predictor.tf.keras.models.load_model') as mock_load:
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([[0.8]]))
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.output_shape = (None, 1)
        mock_load.return_value = mock_model
        
        predictor = MalariaPredictor(model_path="/fake/path/model.h5")
        
        # Create fake image
        image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        result = predictor.predict(image)
        
        assert 'class_name' in result
        assert 'class_id' in result
        assert 'confidence' in result
        assert result['class_name'] in ['Infected', 'Uninfected']


def test_predictor_predict_batch():
    """Test prediction on batch of images"""
    with patch('src.models.predictor.tf.keras.models.load_model') as mock_load:
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([[0.8], [0.2]]))
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.output_shape = (None, 1)
        mock_load.return_value = mock_model
        
        predictor = MalariaPredictor(model_path="/fake/path/model.h5")
        
        # Create fake batch
        images = np.random.randint(0, 256, (2, 224, 224, 3), dtype=np.uint8)
        
        results = predictor.predict_batch(images)
        
        assert len(results) == 2
        assert all('class_name' in r for r in results)
        assert all('confidence' in r for r in results)
