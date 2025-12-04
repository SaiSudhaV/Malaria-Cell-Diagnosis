#!/usr/bin/env python3
"""Environment setup script"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return None

def setup_dvc():
    """Initialize DVC"""
    if not Path('.dvc').exists():
        run_command('dvc init', 'Initializing DVC')
        run_command('dvc config core.analytics false', 'Disabling DVC analytics')
    else:
        print("✅ DVC already initialized")

def setup_prefect():
    """Setup Prefect"""
    run_command('prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api', 'Setting Prefect API URL')
    
def setup_mlflow():
    """Setup MLflow"""
    os.makedirs('mlruns', exist_ok=True)
    print("✅ MLflow directory created")

def setup_directories():
    """Create necessary directories"""
    dirs = [
        'data/raw', 'data/processed', 'data/test_samples',
        'models', 'results', 'metrics', 'predictions',
        'logs'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Project directories created")

def create_env_file():
    """Create .env file with default settings"""
    env_content = """# MLOps Environment Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
PREFECT_API_URL=http://127.0.0.1:4200/api
TF_CPP_MIN_LOG_LEVEL=2
PYTHONUNBUFFERED=1
"""
    
    if not Path('.env').exists():
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ .env file created")
    else:
        print("✅ .env file already exists")

def main():
    """Main setup function"""
    print("🚀 Setting up MLOps environment...")
    
    # Check if we're in the right directory
    if not Path('requirements.txt').exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    setup_directories()
    create_env_file()
    setup_dvc()
    setup_prefect()
    setup_mlflow()
    
    print("\n🎉 Environment setup completed!")
    print("\nNext steps:")
    print("1. Copy your trained model to models/malaria_vgg16_final.h5")
    print("2. Add your dataset to data/raw/")
    print("3. Run: python scripts/preprocess_data.py")
    print("4. Start MLflow: mlflow ui --host 0.0.0.0 --port 5000")
    print("5. Start Prefect: prefect server start")
    print("6. Run workflows: python scripts/run_prefect_workflows.py")

if __name__ == "__main__":
    main()