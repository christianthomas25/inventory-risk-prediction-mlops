# Inventory Risk Prediction – MLOps Pipeline

## Overview

This project implements an end-to-end MLOps pipeline for predicting inventory risk using machine learning. The pipeline covers the full lifecycle from data exploration to deployment and monitoring, ensuring reproducibility, scalability, and reliability.

The system predicts inventory risk levels (e.g., High Risk, Medium Risk, Low Risk) using historical sales and operational data.

---

## Project Structure

```text
01-initial-notebook/        # Stage 1: Data exploration
02-features-modelling/     # Stage 2: Feature engineering
03-experiment-tracking/    # Stage 3: MLflow experiments
04-deployment/             # Stage 4: FastAPI model serving
05-monitoring/             # Stage 5: Monitoring & drift detection
06-cicd/                   # Stage 6: Docker & CI/CD
```

---

# Stage 1: Data Exploration

## Objective

Understand the dataset and identify key variables affecting inventory risk.

## Key Tasks

* Loaded and inspected dataset
* Performed summary statistics
* Visualized distributions (boxplots, histograms)
* Identified potential outliers and anomalies

## Outcome

* Clear understanding of data structure
* Identified important variables such as:

  * Inventory Level
  * Units Sold
  * Demand Forecast
  * Price and Discount

---

# Stage 2: Feature Engineering & Modelling

## Objective

Prepare features and build predictive models.

## Key Tasks

* Cleaned dataset and handled missing values
* Removed leakage variables
* Created new features:

  * Inventory_Reconstructed
  * Inventory_Change_Pct
  * Days_of_Stock
  * Sales_Velocity
* Split data into train / validation / test sets

## Models Implemented

* Logistic Regression
* Random Forest
* XGBoost

## Techniques Used

* One-hot encoding for categorical variables
* Standardization for numerical variables
* Handling class imbalance using SMOTE / SMOTENC

## Evaluation Metrics

* Accuracy
* Precision (macro)
* Recall (macro)
* F1-score (macro)

## Outcome

* Selected best-performing model based on validation performance

---

# Stage 3: Experiment Tracking (MLflow)

## Objective

Track experiments and compare models systematically.

## Key Tasks

* Logged model parameters and metrics using MLflow
* Stored artifacts:

  * Confusion matrices
  * Classification reports
* Compared multiple model runs

## Output

* Best model identified using MLflow runs
* Saved:

  * `run_id.txt`
  * `best_model_uri.txt`

## Outcome

* Reproducible experiment tracking
* Transparent model selection process

---

# Stage 4: Model Deployment (FastAPI)

## Objective

Serve the trained model via an API.

## Key Components

* `app.py` → FastAPI application
* `/predict` endpoint → returns predictions
* `/health` endpoint → service status

## Features

* Input validation using Pydantic
* Data preprocessing aligned with training pipeline
* Support for batch and single predictions

## Testing

* Created `test_api.py`
* Validated API responses using pytest

## Outcome

* Fully functional prediction API

---

# Stage 5: Monitoring & Drift Detection

## Objective

Monitor model performance and detect data drift.

## Key Components

* `simulate.py` → generates prediction data
* `monitor.py` → basic monitoring metrics
* `evidently_report.py` → drift detection

## Outputs

* `predictions.csv` → logged predictions
* `monitoring_report.html` → performance summary
* `evidently_report.html` → drift analysis

## Outcome

* Ability to track model behavior over time
* Early detection of data drift

---

# Stage 6: Docker & CI/CD Pipeline

## Objective

Automate deployment and testing using Docker and GitHub Actions.

---

## Dockerization

### Components

* `app.py`
* `models/model.pkl`
* `models/label_classes.json`
* `Dockerfile`

### Build Image

```bash
docker build -t inventory-risk-api ./06-cicd
```

### Run Container

```bash
docker run -p 5001:8000 inventory-risk-api
```

### Access API

* Swagger UI: http://127.0.0.1:5001/docs
* Health: http://127.0.0.1:5001/health

---

## CI/CD Pipeline (GitHub Actions)

### Workflow Files

```text
.github/workflows/
├── train.yml
└── ci-cd.yml
```

---

### Step 1: Training (`train.yml`)

* Train models
* Select best model
* Save artifacts:

  * `model.pkl`
  * `label_classes.json`
* Upload artifacts

---

### Step 2: Build & Test (`ci-cd.yml`)

Triggered on every push to `main`.

Pipeline:

```text
Train → Download Artifacts → Build Docker → Run API → Run Tests
```

### Automated Validation

* Docker image built successfully
* API container started
* `pytest` validates `/predict` endpoint

---

## Key Design Decisions

### 1. Separation of Training & Deployment

* MLflow used for experimentation only
* Deployment uses local artifacts (`model.pkl`)

### 2. Reproducibility

* Docker ensures consistent environment
* Same setup locally and in CI

### 3. Automation

* Full pipeline runs automatically on push
* Reduces manual errors

---

## Final Outcome

This project delivers a complete MLOps pipeline that includes:

* Data exploration and feature engineering
* Model training and evaluation
* Experiment tracking with MLflow
* API deployment using FastAPI
* Monitoring and drift detection
* Containerization with Docker
* Automated CI/CD pipeline

## Live API

The deployed API is available at:

[https://your-app-name.onrender.com/docs](https://inventory-risk-api.onrender.com/docs#/)

Example endpoints:
- /health
- /predict

---

## Conclusion

The system is production-ready and demonstrates best practices in MLOps, including:

* Reproducibility
* Scalability
* Automation
* Continuous validation

---

## Authors

* Group Project – MLOps Course
* IRP Section 1 Group 5

---
