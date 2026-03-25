# Stage 6: Docker & CI/CD Pipeline

## Overview

In this stage, we containerized our machine learning model using Docker and implemented a Continuous Integration / Continuous Deployment (CI/CD) pipeline using GitHub Actions.

The goal is to ensure that:

* The model can be deployed consistently across environments
* The entire pipeline (training → deployment → testing) is automated
* Any code change is validated before deployment

---

## 1. Dockerized Model Serving

We packaged the FastAPI application into a Docker container.

### Key Components

* `app.py` → FastAPI application for serving predictions
* `models/model.pkl` → trained model (scikit-learn pipeline)
* `models/label_classes.json` → class label mapping
* `Dockerfile` → defines the container setup
* `requirements.txt` → dependencies

### Dockerfile Summary

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 2. Running the API Locally with Docker

### Build Docker Image

```bash
docker build -t inventory-risk-api .
```

### Run Container

```bash
docker run -p 5001:8000 inventory-risk-api
```

### Access API

* Swagger UI:
  http://127.0.0.1:5001/docs

* Health Check:
  http://127.0.0.1:5001/health

---

## 3. API Testing

We use `pytest` to validate the API.

### Run Tests

```bash
pytest test_api.py -v
```

This ensures:

* The `/predict` endpoint works correctly
* The API returns valid predictions

---

## 4. CI/CD Pipeline (GitHub Actions)

We implemented a two-stage pipeline:

### Workflow Files

```text
.github/workflows/
├── train.yml
└── ci-cd.yml
```

---

### Step 1: Model Training (`train.yml`)

This workflow:

* Loads training data
* Trains multiple models
* Selects the best model
* Saves model artifacts
* Uploads artifacts for reuse

Artifacts:

* `models/model.pkl`
* `models/label_classes.json`

---

### Step 2: Build & Test (`ci-cd.yml`)

Triggered on every push to `main`.

Pipeline steps:

1. Call training workflow
2. Download trained model artifacts
3. Build Docker image
4. Run container
5. Execute API tests (`pytest`)

---

### Pipeline Flow

```text
Train Model → Save Artifacts → Build Docker → Run API → Run Tests
```

---

## 5. Key Design Decisions

### 1. No MLflow in Production

* MLflow is used only for experimentation (Stage 3)
* Deployment uses local artifacts (`model.pkl`)
* This ensures portability and avoids runtime dependencies

### 2. End-to-End Automation

* Model training, deployment, and testing are fully automated
* Reduces human error and improves reproducibility

### 3. Reproducibility

* Docker ensures consistent environments
* Same container runs locally and in CI

---

## 6. Results

The CI/CD pipeline successfully:

* Trains the model
* Builds the Docker container
* Deploys the API
* Validates endpoints using automated tests

Total pipeline runtime: ~3–5 minutes

---

## 7. Conclusion

Stage 6 completes the MLOps lifecycle by integrating:

* Model training
* Containerized deployment
* Automated testing
* CI/CD pipeline

This ensures that the machine learning system is:

* Scalable
* Reproducible
* Production-ready

---
