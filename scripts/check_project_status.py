#!/usr/bin/env python3
"""Project status checker"""

import os
import subprocess
from pathlib import Path
import json

def check_file_exists(filepath, description):
    """Check if a file exists"""
    exists = Path(filepath).exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists

def check_service_running(command, service_name):
    """Check if a service is running"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=5)
        running = result.returncode == 0
        status = "✅" if running else "❌"
        print(f"{status} {service_name}")
        return running
    except:
        print(f"❌ {service_name} (timeout/error)")
        return False

def check_project_status():
    """Check complete project status"""
    print("🔍 MLOps Project Status Check")
    print("=" * 50)
    
    # Core Files
    print("\n📁 Core Project Files:")
    core_files = [
        ("README.md", "Project Documentation"),
        ("requirements.txt", "Python Dependencies"),
        ("setup.py", "Package Setup"),
        ("Makefile", "Build Commands"),
        (".gitignore", "Git Ignore Rules")
    ]
    
    for filepath, desc in core_files:
        check_file_exists(filepath, desc)
    
    # Configuration Files
    print("\n⚙️ Configuration Files:")
    config_files = [
        ("config/config.yaml", "Main Configuration"),
        ("dvc.yaml", "DVC Pipeline"),
        ("prefect.yaml", "Prefect Deployment"),
        (".env", "Environment Variables")
    ]
    
    for filepath, desc in config_files:
        check_file_exists(filepath, desc)
    
    # Source Code
    print("\n💻 Source Code:")
    src_files = [
        ("src/data/loader.py", "Data Loader"),
        ("src/models/predictor.py", "Model Predictor"),
        ("api/main.py", "FastAPI Application"),
        ("workflows/prefect_flows.py", "Prefect Workflows")
    ]
    
    for filepath, desc in src_files:
        check_file_exists(filepath, desc)
    
    # Docker Files
    print("\n🐳 Docker Configuration:")
    docker_files = [
        ("docker/Dockerfile", "Production Dockerfile"),
        ("docker-compose.yml", "Docker Compose")
    ]
    
    for filepath, desc in docker_files:
        check_file_exists(filepath, desc)
    
    # CI/CD
    print("\n🔄 CI/CD Configuration:")
    cicd_files = [
        (".github/workflows/ci-cd.yml", "GitHub Actions Workflow")
    ]
    
    for filepath, desc in cicd_files:
        check_file_exists(filepath, desc)
    
    # Data Directories
    print("\n📊 Data Structure:")
    data_dirs = [
        ("data/", "Data Directory"),
        ("models/", "Models Directory"),
        ("tests/", "Tests Directory")
    ]
    
    for dirpath, desc in data_dirs:
        check_file_exists(dirpath, desc)
    
    # Model Files
    print("\n🤖 Model Files:")
    model_files = [
        ("models/malaria_vgg16_final.h5", "Trained Model"),
        ("colab_artifacts/malaria_vgg16_final.h5", "Colab Model Backup")
    ]
    
    for filepath, desc in model_files:
        check_file_exists(filepath, desc)
    
    # Services Status
    print("\n🌐 Services Status:")
    services = [
        ("curl -s http://localhost:5000 > /dev/null", "MLflow Server (port 5000)"),
        ("curl -s http://localhost:8000/health > /dev/null", "FastAPI Server (port 8000)"),
        ("curl -s http://localhost:4200 > /dev/null", "Prefect Server (port 4200)")
    ]
    
    for command, desc in services:
        check_service_running(command, desc)
    
    # Python Environment
    print("\n🐍 Python Environment:")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow: Not installed")
    
    try:
        import prefect
        print(f"✅ Prefect: {prefect.__version__}")
    except ImportError:
        print("❌ Prefect: Not installed")
    
    try:
        import mlflow
        print(f"✅ MLflow: {mlflow.__version__}")
    except ImportError:
        print("❌ MLflow: Not installed")
    
    # Summary
    print("\n📋 Quick Start Commands:")
    print("1. Setup: make setup")
    print("2. Install: make install")
    print("3. Start MLflow: make mlflow-start")
    print("4. Start API: make api-start")
    print("5. Run Tests: make test")
    print("6. Run Workflows: make run-workflow WORKFLOW=preprocessing")
    print("7. Simulate CI/CD: python scripts/simulate_cicd.py")

if __name__ == "__main__":
    check_project_status()