"""FastAPI application for malaria diagnosis inference"""

import logging
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional
import cv2
import io
from pathlib import Path

from src.models.predictor import MalariaPredictor
from src.utils.config import Config
from src.utils.logging_config import setup_logging

# Setup logging
logger = setup_logging("malaria_api")

# Global predictor
predictor = None


# Pydantic models
class PredictionResponse(BaseModel):
    """Single prediction response"""
    class_name: str = Field(..., description="Predicted class name")
    class_id: int = Field(..., description="Predicted class ID")
    raw_prediction: float = Field(..., description="Raw model prediction value")
    confidence: float = Field(..., description="Confidence score")
    threshold: float = Field(..., description="Decision threshold used")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_images: int = Field(..., description="Total images processed")
    processing_time_seconds: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_name: str
    model_version: str


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_name: str
    version: str
    input_shape: tuple
    output_shape: tuple
    classes: List[str]
    threshold: float


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    global predictor
    # Startup
    try:
        logger.info("Loading malaria diagnosis model...")
        predictor = MalariaPredictor()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API")


# Create FastAPI app
app = FastAPI(
    title="Malaria Cell Diagnosis API",
    description="REST API for malaria cell diagnosis using VGG16 transfer learning",
    version="1.0.0",
    lifespan=lifespan
)


# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        model_name=Config.MODEL_NAME,
        model_version=Config.MODEL_VERSION
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = predictor.get_model_info()
    return ModelInfoResponse(
        model_name=info['model_name'],
        version=info['version'],
        input_shape=info['input_shape'],
        output_shape=info['output_shape'],
        classes=info['classes'],
        threshold=info['threshold']
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Make prediction on a single image
    
    Args:
        file: Image file (JPG, PNG, etc.)
        
    Returns:
        Prediction results
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Predict
        result = predictor.predict(image)
        
        return PredictionResponse(
            class_name=result['class_name'],
            class_id=result['class_id'],
            raw_prediction=result['raw_prediction'],
            confidence=result['confidence'],
            threshold=result['threshold']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(files: List[UploadFile] = File(...)):
    """Make predictions on multiple images
    
    Args:
        files: List of image files
        
    Returns:
        Batch prediction results
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        images = []
        for file in files:
            try:
                contents = await file.read()
                nparr = np.frombuffer(contents, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                else:
                    logger.warning(f"Invalid image: {file.filename}")
                    
            except Exception as e:
                logger.warning(f"Error processing {file.filename}: {str(e)}")
                continue
        
        if not images:
            raise HTTPException(status_code=400, detail="No valid images provided")
        
        # Predict batch
        images_array = np.array(images)
        results = predictor.predict_batch(images_array)
        
        processing_time = time.time() - start_time
        
        prediction_responses = [
            PredictionResponse(
                class_name=r['class_name'],
                class_id=r['class_id'],
                raw_prediction=r['raw_prediction'],
                confidence=r['confidence'],
                threshold=r['threshold']
            )
            for r in results
        ]
        
        return BatchPredictionResponse(
            predictions=prediction_responses,
            total_images=len(results),
            processing_time_seconds=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Malaria Cell Diagnosis API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT, log_level="info")
