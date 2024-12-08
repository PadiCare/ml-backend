# ml-backend

# Overview
This repository contains the backend code for PadiCare, an AI-powered rice disease detection application. The backend processes images, runs predictions using machine learning models, and returns the predicted rice disease label. The model is deployed using Flask and integrates with Google Cloud Storage for model and image handling.

# About This Repository
In this repositorym there are some files to support our project:
|  Files |  Purpose |
|--------|--------|
| __pycache__/ | Cached Python files |
| .gitignore | Files and directories to ignore in version control |
| Dockerfile | Docker configuration for deploying the backend |
| README.md | Project documentation |
| app.py | Main Flask application file |
| config.json | Configuration file for Google Cloud Storage buckets | 
| converted_model.tflite | TensorFlow Lite version of the trained model |
| disease_info.json | JSON file with detailed disease information |
| model_padicare.h5 | Trained model in HDF5 format |
| requirements.txt | Dependencies required to run the application | 
