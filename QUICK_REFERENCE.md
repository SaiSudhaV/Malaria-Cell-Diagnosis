# MLOps Quick Reference Card

## 🚀 Essential Commands

### Setup (First Time)
```bash
# Clone & enter directory
cd Malaria-Cell-Diagnosis

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install packages
pip install -r requirements.txt
pip install -e .

# Initialize project
python config/create_configs.py
```

### Start Services
```bash
# Terminal 1: MLflow
mlflow server --host 0.0.0.0 --port 5000

# Terminal 2: API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 3: Workflows (optional)
prefect server start
```

### Key Workflows
```bash
# Register model
python scripts/register_model_mlflow.py

# Evaluate model
python scripts/evaluate.py --split test --mlflow

# Batch predictions
python scripts/batch_predict.py --directory data/processed/test

# Run Prefect flows
python workflows/prefect_flows.py

# Run tests
pytest tests/ -v --cov=src --cov=api
```

### Docker
```bash
# Build image
docker build -f docker/Dockerfile -t malaria-diagnosis:latest .

# Run compose
docker-compose up -d
docker-compose down

# Check services
docker-compose ps
docker-compose logs -f api
```

---

## 📍 Port Reference

| Service | Port | URL |
|---------|------|-----|
| API | 8000 | http://localhost:8000 |
| API Docs | 8000 | http://localhost:8000/docs |
| MLflow | 5000 | http://localhost:5000 |
| Prefect | 4200 | http://localhost:4200 |

---

## 🔌 API Endpoints

### Health & Info
```bash
GET  /health              # Health check
GET  /model-info          # Model details
GET  /                    # API info
```

### Predictions
```bash
POST /predict             # Single image prediction
POST /batch-predict       # Multiple images
```

### Example Requests
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg"

# Batch prediction
curl -X POST "http://localhost:8000/batch-predict" \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg"
```

---

## 📂 Directory Structure

```
Key Directories:
├── data/processed/        Train/val/test splits
├── models/               Your trained model (H5)
├── src/                  Source code (data, models, utils)
├── api/                  FastAPI application
├── scripts/              Utilities (evaluate, predict, register)
├── workflows/            Prefect flows + monitoring
├── tests/                Unit tests
├── docker/               Dockerfiles
├── config/               Configuration files
└── notebooks/            Jupyter notebooks
```

---

## 🧪 Testing

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov=api --cov-report=html

# Specific test file
pytest tests/test_api.py -v

# Specific test
pytest tests/test_api.py::test_health_endpoint -v
```

---

## 🛠️ Configuration

### Model Settings
```python
from src.utils.config import Config
Config.IMG_SIZE = (224, 224)
Config.BATCH_SIZE = 32
Config.PREDICTION_THRESHOLD = 0.5
Config.CLASSES = ["Uninfected", "Infected"]
```

### Modify config.yaml
```yaml
model:
  threshold: 0.5
  
api:
  port: 8000
  debug: false

mlflow:
  tracking_uri: http://localhost:5000
```

---

## 🔍 Debugging

### Check Model Loading
```bash
python -c "from src.models.predictor import MalariaPredictor; \
  p = MalariaPredictor(); print(p.get_model_info())"
```

### Check Data Loading
```bash
python -c "from src.data.loader import ImageDataLoader; \
  loader = ImageDataLoader(); \
  paths = loader.list_images_in_directory('data/raw'); \
  print(f'Found {len(paths)} images')"
```

### Check API Health
```bash
curl http://localhost:8000/health
```

### View Logs
```bash
# API logs
docker-compose logs -f api

# Application logs
tail -f logs/malaria_diagnosis.log
```

### Kill Process on Port
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

---

## 📊 MLflow

### View Experiments
```bash
# Visit http://localhost:5000
# Or use CLI
mlflow runs list --experiment-name malaria-diagnosis
```

### Compare Runs
```bash
mlflow runs compare <run_id1> <run_id2>
```

---

## 🚨 Common Issues

### "Model not found"
```bash
# Check model exists
ls -la models/malaria_vgg16_final.h5

# Update config if needed
# config/config.yaml: model_path: models/malaria_vgg16_final.h5
```

### "Port already in use"
```bash
# Use different port
python -m uvicorn api.main:app --port 8001
```

### "Import errors"
```bash
# Reinstall in editable mode
pip install -e .
```

### "Data not found"
```bash
# Verify data structure
data/raw/
├── Uninfected/
└── Infected/

# Check DVC pull if using DVC
dvc pull
```

### "Docker issues"
```bash
# Clean and rebuild
docker system prune -a
docker-compose build --no-cache
```

---

## 📝 Useful Python Snippets

### Test Single Prediction
```python
import cv2
from src.models.predictor import MalariaPredictor

predictor = MalariaPredictor()
image = cv2.imread('path/to/image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = predictor.predict(image)
print(result)
```

### Load and Process Batch
```python
from src.data.loader import ImageDataLoader
from src.data.preprocessor import ImagePreprocessor

loader = ImageDataLoader()
preprocessor = ImagePreprocessor()

paths, labels = loader.load_split_data('test')
images = loader.load_batch(paths[:10])
processed = preprocessor.preprocess_batch(images)
print(f'Processed shape: {processed.shape}')
```

### Log to MLflow
```python
import mlflow
from scripts.register_model_mlflow import MLflowManager

manager = MLflowManager()
manager.start_run("test_run")
manager.log_params({'lr': 0.001})
manager.log_metrics({'accuracy': 0.95})
manager.end_run()
```

### Run Workflow
```python
from workflows.prefect_flows import evaluation_flow

metrics = evaluation_flow('data/processed', 'test')
print(metrics)
```

---

## 🔐 Production Checklist

- [ ] Model tested and validated
- [ ] Data splits created (80/10/10)
- [ ] API endpoints tested
- [ ] Unit tests passing
- [ ] Docker image built
- [ ] MLflow tracking working
- [ ] Prefect workflows running
- [ ] Monitoring enabled
- [ ] CI/CD pipeline active
- [ ] Documentation complete
- [ ] README updated
- [ ] All commits pushed

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| README.md | Main documentation |
| IMPLEMENTATION_GUIDE.md | Step-by-step setup |
| COMPLETION_SUMMARY.md | What's included |
| Quick Reference (this) | Commands & tips |

---

## 🎯 Next Steps

1. **Prepare data** → Copy dataset to `data/raw/`
2. **Copy model** → Place H5 file in `models/`
3. **Test setup** → Run `pytest tests/ -v`
4. **Start services** → Run MLflow + API
5. **Register model** → `python scripts/register_model_mlflow.py`
6. **Test API** → Visit http://localhost:8000/docs
7. **Run workflows** → `python workflows/prefect_flows.py`
8. **Deploy** → `docker-compose up -d`
9. **Push code** → `git push origin main`
10. **Create video** → Record 5-10 min demo

---

**Save this file as a bookmark or print it!** 📌

