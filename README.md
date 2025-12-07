# Malaria Cell Diagnosis - MLOps Pipeline

A complete end-to-end MLOps project for malaria cell diagnosis using transfer learning with VGG16. This project demonstrates all components of a production-ready ML pipeline using open-source tools running locally.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [MLOps Components](#mlops-components)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Usage](#api-usage)
- [Model Training & Evaluation](#model-training--evaluation)
- [Workflow Orchestration](#workflow-orchestration)
- [Docker Deployment](#docker-deployment)
- [Monitoring & Tracking](#monitoring--tracking)
- [CI/CD Pipeline](#cicd-pipeline)
- [Testing](#testing)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## 🎯 Project Overview

### Problem Statement
This project builds a machine learning pipeline to diagnose malaria from cell images using a pre-trained VGG16 model. The focus is on implementing production-grade MLOps practices including:

- **Data Management**: Version control with DVC
- **Model Registry**: MLflow for experiment tracking and model versioning
- **Workflow Orchestration**: Prefect for pipeline orchestration
- **API Deployment**: FastAPI for inference endpoints
- **Containerization**: Docker for reproducible deployments
- **Monitoring**: Data drift and model performance monitoring
- **CI/CD**: GitHub Actions for automated testing and deployment

### Dataset
- **Source**: Malaria Cell Images dataset
- **Task**: Binary classification (Infected vs Uninfected)
- **Split**: 80% train, 10% validation, 10% test
- **Model**: VGG16 Transfer Learning (trained on ImageNet)

## 🛠️ MLOps Components

| Component | Tool | Purpose |
|-----------|------|---------|
| **Data Versioning** | DVC | Track and version control datasets |
| **Model Registry** | MLflow | Experiment tracking, model registration, versioning |
| **Workflow Orchestration** | Prefect | Schedule and orchestrate ML workflows |
| **API Framework** | FastAPI | Build REST API for inference |
| **Containerization** | Docker | Package application with dependencies |
| **Monitoring** | Evidently AI | Data drift and model performance monitoring |
| **CI/CD** | GitHub Actions | Automated testing and deployment |
| **Testing** | Pytest | Unit and integration tests |

## 📁 Project Structure

```
Malaria-Cell-Diagnosis/
├── data/                           # Data directory
│   ├── raw/                        # Original dataset
│   ├── processed/                  # Processed train/val/test splits
│   └── test_samples/               # Sample images for testing
├── models/                         # Model artifacts
│   ├── malaria_vgg16_final.h5     # Trained model
│   ├── preprocessing/              # Preprocessing models
│   └── saved_model/                # TF SavedModel format
├── src/                            # Source code
│   ├── data/                       # Data loading & preprocessing
│   │   ├── loader.py              # ImageDataLoader class
│   │   └── preprocessor.py        # ImagePreprocessor class
│   ├── models/                     # Model inference
│   │   └── predictor.py           # MalariaPredictor class
│   └── utils/                      # Utilities
│       ├── config.py              # Configuration management
│       └── logging_config.py      # Logging setup
├── api/                            # FastAPI application
│   └── main.py                     # API routes and endpoints
├── scripts/                        # Standalone scripts
│   ├── register_model_mlflow.py   # Register model in MLflow
│   ├── evaluate.py                # Model evaluation
│   └── batch_predict.py           # Batch prediction
├── workflows/                      # Prefect workflows
│   ├── prefect_flows.py           # Workflow definitions
│   └── monitoring.py              # Monitoring utilities
├── notebooks/                      # Jupyter notebooks
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   ├── 02_evaluation.ipynb        # Model evaluation
│   └── 03_inference_demo.ipynb    # Inference demonstration
├── tests/                          # Unit tests
│   ├── test_data.py               # Data loading tests
│   ├── test_model.py              # Model tests
│   └── test_api.py                # API tests
├── config/                         # Configuration files
│   ├── config.yaml                # Main configuration
│   ├── logging.yaml               # Logging configuration
│   └── create_configs.py          # Config creation script
├── docker/                         # Docker files
│   ├── Dockerfile                 # Production Dockerfile
│   └── Dockerfile.dev             # Development Dockerfile
├── .github/workflows/              # CI/CD workflows
│   └── ci-cd.yml                  # GitHub Actions workflow
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── docker-compose.yml             # Docker Compose configuration
└── README.md                       # This file

```

## 📦 Requirements

- Python 3.9+
- Docker & Docker Compose (for containerization)
- Git (for version control)
- At least 8GB RAM
- GPU recommended (CUDA 11.0+) but CPU works fine

## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/SaiSudhaV/Malaria-Cell-Diagnosis.git
cd Malaria-Cell-Diagnosis
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Setup Project
```bash
python src/utils/config.py
python config/create_configs.py
```

### 5. Prepare Data
Place your dataset in the following structure:
```
data/raw/
├── Uninfected/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Infected/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Or use DVC to pull data:
```bash
dvc pull
```

### 6. Copy Trained Model
Copy your `malaria_vgg16_final.h5` from Google Colab to `models/` directory:
```
models/
└── malaria_vgg16_final.h5
```

## ⚡ Quick Start

### 1. Start MLflow Server
```bash
mlflow ui --host 0.0.0.0 --port 5000
# Visit http://localhost:5000
```

### 2. Register Model
```bash
python scripts/register_model_mlflow.py
```

### 3. Start FastAPI Server
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 4. Make a Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/image.jpg"
```

### 5. Run Workflows
```bash
python workflows/prefect_flows.py
# Or use Prefect UI: prefect server start
```

## 🔌 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "malaria_vgg16",
  "model_version": "1.0"
}
```

### Single Image Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg"
```

Response:
```json
{
  "class_name": "Infected",
  "class_id": 1,
  "raw_prediction": 0.8523,
  "confidence": 0.8523,
  "threshold": 0.5
}
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/batch-predict" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

### Interactive API Documentation
Visit http://localhost:8000/docs for Swagger UI documentation

## 📊 Model Training & Evaluation

### Evaluate on Test Set
```bash
python scripts/evaluate.py --split test --mlflow
```

### Generate Evaluation Plots
```bash
python scripts/evaluate.py --split test --output results/eval.json
```

Output includes:
- Confusion matrix
- ROC curve
- Classification report
- Prediction distribution

## 🔄 Workflow Orchestration

### Run Prefect Workflows
```bash
# Data preprocessing flow
python workflows/prefect_flows.py

# Or use Prefect CLI
prefect flow run preprocessing_flow
prefect flow run evaluation_flow
prefect flow run batch_prediction_flow
```

### Monitor Workflows
```bash
prefect server start
# Visit http://localhost:4200
```

## 🐳 Docker Deployment

### Build and Run with Docker Compose
```bash
# Start all services (API, MLflow, Prefect)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

Services:
- **API**: http://localhost:8000
- **MLflow**: http://localhost:5000
- **Prefect**: http://localhost:4200

### Stop Services
```bash
docker-compose down
```

### Build Custom Image
```bash
docker build -f docker/Dockerfile -t malaria-diagnosis:latest .
```

## 📈 Monitoring & Tracking

### MLflow Experiment Tracking
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

View:
- Experiment runs
- Model versions
- Metrics and parameters
- Artifacts

### Model Performance Monitoring
```bash
python workflows/monitoring.py
```

Monitors:
- Prediction confidence distribution
- Data drift detection
- Model performance over time

### Generate Monitoring Report
```python
from workflows.monitoring import ModelMonitor

monitor = ModelMonitor()
report = monitor.generate_report()
print(report)
```

## 🔁 CI/CD Pipeline

GitHub Actions workflow runs on every push to main/develop:

1. **Unit Tests**: Run all pytest tests
2. **Code Linting**: Check code quality
3. **Coverage**: Generate coverage reports
4. **Docker Build**: Build Docker image
5. **Security Scan**: Trivy vulnerability scan

View workflow status:
```
GitHub Repo → Actions → CI/CD Pipeline
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov=api --cov-report=html
# Open htmlcov/index.html
```

### Run Specific Test
```bash
pytest tests/test_api.py::test_health_endpoint -v
```

### Test Categories
- **Data tests**: `tests/test_data.py`
- **Model tests**: `tests/test_model.py`
- **API tests**: `tests/test_api.py`

## ⚙️ Configuration

### config/config.yaml
Main configuration file with:
- Model parameters
- Data settings
- Training hyperparameters
- API configuration
- MLflow settings

### Environment Variables
Create `.env` file:
```
MLFLOW_TRACKING_URI=http://localhost:5000
TF_CPP_MIN_LOG_LEVEL=2
PYTHONUNBUFFERED=1
```

### Modify Settings
```python
from src.utils.config import Config

# Override settings
Config.IMG_SIZE = (256, 256)
Config.BATCH_SIZE = 64
Config.PREDICTION_THRESHOLD = 0.6
```

## 🐛 Troubleshooting

### Model Not Loading
```bash
# Verify model file exists
ls -la models/malaria_vgg16_final.h5

# Check TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

### API Port Already in Use
```bash
# Use different port
python -m uvicorn api.main:app --port 8001
```

### Docker Build Issues
```bash
# Clear Docker cache
docker system prune -a

# Rebuild
docker-compose build --no-cache
```

### Data Loading Errors
```bash
# Verify data structure
python -c "from src.data.loader import ImageDataLoader; loader = ImageDataLoader(); print(loader.list_images_in_directory('data/raw'))"
```

### MLflow Connection Issues
```bash
# Check MLflow server
curl http://localhost:5000

# Start MLflow if not running
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

## 📝 Notebooks

### 01_eda.ipynb
Exploratory Data Analysis
- Dataset statistics
- Class distribution
- Sample images
- Image statistics

### 02_evaluation.ipynb
Model Evaluation
- Load model
- Test set evaluation
- Metrics visualization
- Confusion matrix

### 03_inference_demo.ipynb
Inference Demonstration
- Load model
- Make predictions
- Visualize results
- Batch predictions

---

**Last Updated**: 2025-12-07
**Version**: 1.0.0
