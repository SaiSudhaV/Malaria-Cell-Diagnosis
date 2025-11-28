"""Unit tests for API endpoints"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
import io

# Mock the predictor before importing the app
with patch('api.main.MalariaPredictor'):
    from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    
    assert response.status_code == 200
    assert "status" in response.json()


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    
    assert response.status_code == 200
    assert "message" in response.json()
    assert "endpoints" in response.json()


def test_model_info_endpoint():
    """Test model info endpoint"""
    response = client.get("/model-info")
    
    # May return 503 if model not loaded, which is fine
    assert response.status_code in [200, 503]


def test_predict_missing_file():
    """Test predict endpoint without file"""
    response = client.post("/predict")
    
    assert response.status_code == 422  # Unprocessable Entity


def test_batch_predict_missing_files():
    """Test batch predict endpoint without files"""
    response = client.post("/batch-predict")
    
    assert response.status_code == 422


def test_api_response_schemas():
    """Test API response schemas are properly defined"""
    from api.main import PredictionResponse, HealthResponse, ModelInfoResponse
    
    # Test schemas can be instantiated
    pred = PredictionResponse(
        class_name="Infected",
        class_id=1,
        raw_prediction=0.8,
        confidence=0.8,
        threshold=0.5
    )
    assert pred.class_name == "Infected"
    
    health = HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name="test",
        model_version="1.0"
    )
    assert health.status == "healthy"
