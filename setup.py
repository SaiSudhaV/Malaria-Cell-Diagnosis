from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="malaria-cell-diagnosis",
    version="1.0.0",
    author="Malaria Diagnosis Team",
    description="End-to-end MLOps pipeline for malaria cell diagnosis using VGG16 transfer learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SaiSudhaV/Malaria-Cell-Diagnosis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.14.0",
        "fastapi>=0.103.0",
        "uvicorn>=0.23.2",
        "mlflow>=2.8.0",
        "prefect>=2.13.0",
        "dvc>=3.36.1",
        "evidently>=0.4.16",
        "scikit-learn>=1.3.0",
        "opencv-python>=4.8.0",
        "pydantic>=2.3.0",
    ],
)
