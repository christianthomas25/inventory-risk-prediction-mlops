# Inventory Risk Prediction – End-to-End MLOps Pipeline

## Objective

This project builds a complete end-to-end MLOps pipeline to predict inventory risk levels for retail products. The system is designed to simulate a production-ready machine learning workflow, from data processing to deployment, monitoring, and automation.

The goal is to support better inventory decisions by classifying products into:

- Low Risk  
- Medium Risk  
- High Risk  

---

## Project Overview

The pipeline follows a structured multi-stage approach:

1. Feature Engineering & Data Understanding  
2. Modeling & Experiment Tracking  
3. Deployment (API)  
4. Monitoring  
5. Automation & CI/CD  

Each stage reflects a real-world ML lifecycle component.

---

## Problem Definition

The task is a **multiclass classification problem**:

- Input: Daily inventory and sales features  
- Output: Risk level for future inventory imbalance  

Business objective:

- Reduce stockouts  
- Avoid overstocking  
- Improve operational efficiency  

---

## Pipeline Architecture

**End-to-End Flow:**

1. Raw data → feature engineering  
2. Feature set → model training (MLflow)  
3. Best model → API deployment (FastAPI)  
4. API → real-time predictions  
5. Predictions → logging and monitoring  
6. System → containerized and automated  

---

## Stage 2 – Feature Engineering & Data Understanding

### Key Contributions

- Creation of domain-specific features:
  - Inventory dynamics (e.g., Days_of_Stock)  
  - Demand behavior (Sales_Velocity)  
  - Forecast reliability (Forecast_Error)  

- Target label defined using threshold-based rules  

- Three business scenarios:
  - Conservative  
  - Balanced  
  - Sensitive  

### Sensitivity Analysis

- Evaluates how thresholds impact class distribution  
- Demonstrates that label design depends on business trade-offs  

Important distinction:

- This is **analysis**, not model tuning  

---

## Stage 3 – Modeling & MLflow

### Models Trained

- Logistic Regression  
- Random Forest  
- XGBoost  

### Evaluation Metrics

- Accuracy  
- Precision (macro)  
- Recall (macro)  
- F1-score (macro)  

### MLflow Usage

- Experiment tracking  
- Model comparison  
- Artifact storage  

### Outputs

- `run_id.txt`  
- `best_model_uri.txt`  
- `results_df.csv`  

### Outcome

- Best model selected based on validation performance  
- Fully reproducible training pipeline  

---

## Stage 4 – Deployment (FastAPI)

### API Features

- REST API for real-time predictions  
- Input validation using Pydantic  
- Support for batch and single predictions  

### Endpoints

- `/` → API info  
- `/health` → system status  
- `/predict` → model inference  

### Model Loading Strategy

- Packaged model (portable)  
- MLflow URI fallback  

### Key Achievement

- Model successfully exposed as a production-like service  

---

## Stage 5 – Monitoring

### Monitoring Capabilities

- Prediction logging (inputs + outputs + timestamp)  
- Simulation of production traffic  
- Data quality checks  
- Prediction distribution tracking  
- Data drift detection (Evidently)  

### Outputs

- `prediction_logs.json`  
- `monitoring_summary.csv`  
- `drift_report.html`  

### Purpose

- Ensure model reliability post-deployment  
- Detect anomalies and distribution shifts  

---

## Stage 6 – Automation & CI/CD

### Containerization

- Docker used to package the API  
- Ensures environment consistency and portability  

### CI/CD Pipeline

Implemented with GitHub Actions:

- Install dependencies  
- Run linting checks (flake8)  

### Code Quality

- `.flake8` configuration applied  
- Non-critical warnings ignored  

### Key Challenge Solved

- MLflow artifact paths are not portable  
- Resolved using `packaged_model/`  

---

## Technical Stack

### Core Libraries

- Python  
- pandas, numpy  
- scikit-learn  
- XGBoost  

### MLOps Tools

- MLflow (experiment tracking)  
- FastAPI (deployment)  
- Docker (containerization)  
- GitHub Actions (CI/CD)  
- Evidently (monitoring)  

---

## How to Run the Project

### 1. Train Models
```bash
python train.py
```


### 2. Run API
uvicorn app:app --reload --port 8000
### 3. Simulate Traffic
python simulate.py
### 4. Run Monitoring
python monitor.py
python evidently_report.py
### 5. Run with Docker
docker build -f 06-cicd/Dockerfile -t inventory-api .
docker run -p 8000:8000 inventory-api
Key Design Decisions

#### 1. Label Engineering vs Model Optimization
Labels defined in Stage 2
Model training performed in Stage 3

This separation ensures:

Clear reasoning
Avoidance of data leakage
#### 2. Scenario-Based Thinking
Multiple labeling strategies simulate business priorities
“Best” model depends on operational cost, not just metrics
#### 3. Portability First
Use of packaged_model/ instead of raw MLflow artifacts
Ensures compatibility in Docker environments
#### 4. Observability
Logging and monitoring integrated after deployment
Reflects real-world production requirements
Business Impact

The system enables:

Real-time inventory risk assessment
Better stock management decisions
Reduced operational inefficiencies

Strategic value:

Supports scalable, data-driven decision-making
Bridges the gap between ML models and business operations
Limitations
Dataset is synthetic → may inflate performance
Labels are heuristic-based, not ground truth
Monitoring is simulated, not real-time production
Final Outcome

### The project delivers a complete MLOps pipeline that is:

Reproducible
Modular
Deployable
Monitorable
Production-ready (academic level)
Future Improvements
Real-time streaming data integration
Automated retraining pipeline
Alerting system for drift detection
Use of real-world retail datasets
