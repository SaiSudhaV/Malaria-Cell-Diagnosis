# MLOps Implementation Guide - Complete Workflow

This guide provides step-by-step instructions to implement and complete your entire MLOps pipeline for malaria cell diagnosis.

## 📌 Phase Overview

```
Phase 1: Setup & Installation ──→ Phase 2: Data Preparation ──→ Phase 3: Model Integration
    ↓                                    ↓                           ↓
    └─→ Phase 4: MLOps Components ──→ Phase 5: Testing ──→ Phase 6: Deployment
```

---

## Phase 1: Setup & Installation (30 minutes)

### Step 1.1: Clone and Setup Repository

```bash
# Navigate to workspace
cd c:\workspace\DML_Project\Malaria-Cell-Diagnosis

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')"
```

### Step 1.2: Initialize Project Structure

```bash
# Create necessary directories
mkdir -p data\raw data\processed data\test_samples
mkdir -p models models\preprocessing models\saved_model
mkdir -p logs predictions monitoring\reports monitoring\dashboards

# Create configuration files
python config\create_configs.py
```

### Step 1.3: Setup Git & DVC

```bash
# Initialize Git (if not already done)
git init
git add .
git commit -m "Initial MLOps project structure"

# Initialize DVC
dvc init
dvc remote add -d storage ./data/.dvc_storage

# Add data directories to DVC
dvc add data/raw -r storage
git add data/raw.dvc .gitignore
git commit -m "Add data to DVC"
```

---

## Phase 2: Data Preparation (45 minutes)

### Step 2.1: Organize Your Dataset

Structure your data from Colab like this:

```
data/raw/
├── Uninfected/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ... (hundreds more)
└── Infected/
    ├── image_001.jpg
    ├── image_002.jpg
    └── ... (hundreds more)
```

### Step 2.2: Validate Data

```bash
# Test data loading
python -c "
from src.data.loader import ImageDataLoader
loader = ImageDataLoader('data/raw')
images = loader.list_images_in_directory('data/raw')
print(f'Total images: {len(images)}')
print(f'Sample images: {images[:3]}')
"

# Validate images
python -c "
from src.data.loader import ImageDataLoader
loader = ImageDataLoader('data/raw')
total, valid = loader.validate_images('data/raw')
print(f'Total images: {total}, Valid: {valid}')
"
```

### Step 2.3: Create Data Splits

```bash
# Create train/val/test splits
python -c "
import os
import random
import shutil
from pathlib import Path

SRC = 'data/raw'
TARGET = 'data/processed'
train_pct, val_pct, test_pct = 0.8, 0.1, 0.1

for split in ['train', 'val', 'test']:
    Path(TARGET, split).mkdir(parents=True, exist_ok=True)

for cls in ['Uninfected', 'Infected']:
    src_cls = Path(SRC) / cls
    files = list(src_cls.glob('*.jpg')) + list(src_cls.glob('*.png'))
    random.seed(42)
    random.shuffle(files)
    
    n = len(files)
    n_train = int(n * train_pct)
    n_val = int(n * val_pct)
    
    for split, file_list in [
        ('train', files[:n_train]),
        ('val', files[n_train:n_train+n_val]),
        ('test', files[n_train+n_val:])
    ]:
        for f in file_list:
            dst = Path(TARGET) / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / f.name)

print('Data splits created successfully')
"
```

### Step 2.4: Track Data with DVC

```bash
# Track processed data
dvc add data/processed
git add data/processed.dvc .gitignore
git commit -m "Add processed data splits"

# Push data to DVC remote
dvc push
```

---

## Phase 3: Model Integration (30 minutes)

### Step 3.1: Copy Your Trained Model

Copy the `malaria_vgg16_final.h5` file from Google Colab:

```bash
# Should be at: models/malaria_vgg16_final.h5
ls -la models/malaria_vgg16_final.h5

# Verify model can be loaded
python -c "
from src.models.predictor import MalariaPredictor
predictor = MalariaPredictor('models/malaria_vgg16_final.h5')
info = predictor.get_model_info()
print(f'Model loaded successfully')
print(f'Model info: {info}')
"
```

### Step 3.2: Test Model Inference

```bash
# Test single image prediction
python -c "
import cv2
import numpy as np
from src.models.predictor import MalariaPredictor

# Load predictor
predictor = MalariaPredictor('models/malaria_vgg16_final.h5')

# Create dummy image for testing
image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

# Predict
result = predictor.predict(image)
print(f'Prediction result: {result}')
"
```

### Step 3.3: Test Batch Predictions

```bash
# Test batch prediction
python -c "
import os
from src.data.loader import ImageDataLoader
from src.models.predictor import MalariaPredictor

loader = ImageDataLoader('data/processed')
image_paths, labels = loader.load_split_data('test')

# Predict on first 10 images
test_paths = image_paths[:10]
images = loader.load_batch(test_paths)

predictor = MalariaPredictor('models/malaria_vgg16_final.h5')
results = predictor.predict_batch(images)

for i, r in enumerate(results):
    print(f'Image {i}: {r[\"class_name\"]} ({r[\"confidence\"]:.2%})')
"
```

---

## Phase 4: MLOps Components (1.5 hours)

### Step 4.1: Setup MLflow

```bash
# Start MLflow UI (in separate terminal)
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

# Visit http://localhost:5000
```

### Step 4.2: Register Model in MLflow

```bash
# Register the trained model
python scripts/register_model_mlflow.py

# Verify in MLflow UI at http://localhost:5000
```

### Step 4.3: Start FastAPI Server

```bash
# In new terminal, start API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Visit http://localhost:8000/docs for API documentation
```

### Step 4.4: Test API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model-info

# Single prediction (need an actual image)
# curl -X POST "http://localhost:8000/predict" -F "file=@test_image.jpg"

# Python test
python -c "
import requests
import cv2
import numpy as np
import io
from PIL import Image

# Create test image
img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(img_array)
img_bytes = io.BytesIO()
img.save(img_bytes, format='JPEG')
img_bytes.seek(0)

# Make request
files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
response = requests.post('http://localhost:8000/predict', files=files)
print(f'Status: {response.status_code}')
print(f'Response: {response.json()}')
"
```

### Step 4.5: Setup Prefect Workflows

```bash
# Run preprocessing flow
python -c "
from workflows.prefect_flows import preprocessing_flow
result = preprocessing_flow('data/processed', 'train')
print(f'Preprocessing flow completed')
"

# Run evaluation flow
python -c "
from workflows.prefect_flows import evaluation_flow
metrics = evaluation_flow('data/processed', 'test')
print(f'Evaluation metrics: {metrics}')
"

# Run batch prediction flow
python -c "
from workflows.prefect_flows import batch_prediction_flow
results = batch_prediction_flow('data/processed', 'test')
print(f'Batch prediction completed: {len(results)} predictions')
"

# Start Prefect UI (optional, in new terminal)
prefect server start
# Visit http://localhost:4200
```

### Step 4.6: Model Evaluation

```bash
# Evaluate on test set and generate plots
python scripts/evaluate.py --data-dir data/processed --split test --mlflow --output evaluation_results/test_eval.json

# View results
cat evaluation_results/test_eval.json
```

---

## Phase 5: Testing (30 minutes)

### Step 5.1: Run Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov=api --cov-report=html

# Open htmlcov/index.html to view coverage report
```

### Step 5.2: Test Data Loading

```bash
pytest tests/test_data.py -v
```

### Step 5.3: Test Model Inference

```bash
pytest tests/test_model.py -v
```

### Step 5.4: Test API Endpoints

```bash
pytest tests/test_api.py -v
```

---

## Phase 6: Deployment (1 hour)

### Step 6.1: Build Docker Image

```bash
# Build production image
docker build -f docker/Dockerfile -t malaria-diagnosis:latest .

# Build development image
docker build -f docker/Dockerfile.dev -t malaria-diagnosis:dev .

# Verify image
docker images | grep malaria
```

### Step 6.2: Run with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Test API
curl http://localhost:8000/health
curl http://localhost:5000  # MLflow

# Stop services
docker-compose down
```

### Step 6.3: Local Deployment Testing

```bash
# Run container directly
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  malaria-diagnosis:latest

# Test from another terminal
curl http://localhost:8000/health
```

---

## Phase 7: Batch Processing & Monitoring (45 minutes)

### Step 7.1: Run Batch Predictions

```bash
# Batch predict on all images in a directory
python scripts/batch_predict.py --directory data/processed/test

# With output directory
python scripts/batch_predict.py \
  --directory data/processed/test \
  --output predictions/batch_1 \
  --monitor
```

### Step 7.2: Monitor Model Performance

```bash
# Generate monitoring report
python -c "
from workflows.monitoring import ModelMonitor
monitor = ModelMonitor()
report = monitor.generate_report()
print(f'Total predictions: {report.get(\"total_predictions\", 0)}')
print(f'Predictions by class: {report.get(\"predictions_by_class\", {})}')
"
```

### Step 7.3: Check Data Drift

```bash
# Analyze data drift
python -c "
from workflows.monitoring import ModelMonitor
import json

monitor = ModelMonitor()
predictions_file = 'monitoring/reports/predictions_log.csv'

# Generate drift report
import pandas as pd
df = pd.read_csv(predictions_file)
recent_predictions = df.tail(100).to_dict('records')

drift_report = monitor.check_data_drift(recent_predictions)
print(f'Data drift analysis:')
print(json.dumps(drift_report, indent=2))
"
```

---

## Phase 8: Final Integration & Verification (30 minutes)

### Step 8.1: Git Commits and Push

```bash
# Add all changes
git add .

# Commit with message
git commit -m "Complete MLOps pipeline implementation

- Data loading and preprocessing utilities
- Model inference wrapper
- FastAPI REST API
- MLflow experiment tracking
- Prefect workflow orchestration
- Docker containerization
- Unit tests
- GitHub Actions CI/CD
- Monitoring utilities
- Comprehensive documentation"

# Push to GitHub
git push origin main
```

### Step 8.2: Verify All Components

```bash
# Checklist verification script
python -c "
import os
from pathlib import Path

checks = {
    'Model file': 'models/malaria_vgg16_final.h5',
    'Data directory': 'data/processed',
    'API main': 'api/main.py',
    'Predictor module': 'src/models/predictor.py',
    'MLflow script': 'scripts/register_model_mlflow.py',
    'Prefect flows': 'workflows/prefect_flows.py',
    'Tests': 'tests/test_api.py',
    'Dockerfile': 'docker/Dockerfile',
    'README': 'README.md',
    'Config': 'config/config.yaml',
}

print('\\n' + '='*60)
print('MLOps COMPONENTS VERIFICATION')
print('='*60)

all_good = True
for name, path in checks.items():
    exists = os.path.exists(path) or Path(path).exists()
    status = '✓' if exists else '✗'
    print(f'{status} {name}: {path}')
    if not exists:
        all_good = False

print('='*60)
print(f'Status: {\"ALL COMPONENTS READY\" if all_good else \"MISSING COMPONENTS\"}')"
```

### Step 8.3: Create Quick Start Guide

```bash
# Create a quick start script
cat > run.sh << 'EOF'
#!/bin/bash

echo "MLOps Pipeline Quick Start"
echo "============================"

echo "1. Starting MLflow..."
mlflow server --host 0.0.0.0 --port 5000 &
sleep 3

echo "2. Registering model..."
python scripts/register_model_mlflow.py

echo "3. Starting API..."
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
sleep 3

echo "4. Running evaluation..."
python scripts/evaluate.py --data-dir data/processed --split test --mlflow

echo "5. Running batch predictions..."
python scripts/batch_predict.py --directory data/processed/test --monitor

echo ""
echo "MLOps Pipeline is running!"
echo "================================"
echo "API: http://localhost:8000/docs"
echo "MLflow: http://localhost:5000"
echo "================================"
EOF

chmod +x run.sh
```

---

## 🚀 Quick Command Reference

### Development
```bash
# Activate environment
venv\Scripts\activate

# Run tests
pytest tests/ -v

# Start API
python -m uvicorn api.main:app --reload

# Run workflows
python workflows/prefect_flows.py

# Evaluate model
python scripts/evaluate.py --split test --mlflow
```

### Deployment
```bash
# Docker compose
docker-compose up -d
docker-compose down

# Docker CLI
docker build -f docker/Dockerfile -t malaria-diagnosis:latest .
docker run -p 8000:8000 malaria-diagnosis:latest

# Batch predict
python scripts/batch_predict.py --directory data/processed/test
```

### Monitoring
```bash
# MLflow UI
mlflow ui --port 5000

# Prefect UI
prefect server start

# View logs
docker-compose logs -f api
tail -f logs/malaria_diagnosis.log
```

---

## 📋 Evaluation Checklist

Use this to track your progress:

- [ ] **Problem Definition (5 marks)**
  - [ ] Clear problem statement
  - [ ] Dataset description
  - [ ] Model architecture explanation

- [ ] **EDA (10 marks)**
  - [ ] Dataset statistics
  - [ ] Class distribution analysis
  - [ ] Sample visualizations

- [ ] **Data Preprocessing & DVC (10 marks)**
  - [ ] Train/val/test splits created
  - [ ] Data validation
  - [ ] DVC tracking configured

- [ ] **Model & MLflow (15 marks)**
  - [ ] Model loads successfully
  - [ ] Predictions working
  - [ ] MLflow tracking running
  - [ ] Model registered in MLflow

- [ ] **Prefect Workflow (15 marks)**
  - [ ] Preprocessing flow
  - [ ] Evaluation flow
  - [ ] Batch prediction flow
  - [ ] All flows running without errors

- [ ] **CI/CD (10 marks)**
  - [ ] GitHub Actions workflow defined
  - [ ] Tests passing
  - [ ] Docker build passing

- [ ] **Docker (10 marks)**
  - [ ] Dockerfile working
  - [ ] docker-compose.yml configured
  - [ ] All services running

- [ ] **Deployment (10 marks)**
  - [ ] API endpoints working
  - [ ] Health checks passing
  - [ ] Inference working end-to-end

- [ ] **Monitoring (10 marks)**
  - [ ] Prediction logging working
  - [ ] Performance metrics tracked
  - [ ] Data drift detection

- [ ] **Documentation (5 marks)**
  - [ ] README complete
  - [ ] Setup instructions clear
  - [ ] API documentation available

---

## 🎥 Video Demo Outline (5-10 minutes)

1. **Introduction (1 min)**
   - Problem statement
   - Solution approach

2. **Project Structure (1 min)**
   - Directory layout
   - Key components

3. **Live Demo (5 mins)**
   - Start services
   - Make API prediction
   - Show MLflow tracking
   - Run Prefect flow
   - View Docker deployment

4. **Results & Metrics (1-2 mins)**
   - Model performance
   - Evaluation metrics
   - Monitoring dashboard

5. **Conclusion (1 min)**
   - Summary
   - Future improvements

---

**TOTAL IMPLEMENTATION TIME: ~4-5 hours**

After completing all phases, you will have a production-ready MLOps pipeline with all required components!

