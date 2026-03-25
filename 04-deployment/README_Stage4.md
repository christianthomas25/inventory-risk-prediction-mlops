# Stage 4: Model Serving (FastAPI)

## Overview

In this stage, we deploy our trained machine learning model as a web service using **FastAPI**.
The model is exposed through a REST API, allowing external applications to send data and receive predictions in real time.

---

## Pipeline Flow

```id="7w3d2k"
train.py → trains model (MLflow)
        ↓
run_id.txt / best_model_uri.txt
        ↓
app.py → loads model
        ↓
FastAPI (/predict endpoint)
        ↓
Client request → prediction response
```

---

## 📁 Project Structure

```id="v0h3cp"
04-deployment/
│
├── train.py              # Model training script (MLflow)
├── app.py                # FastAPI application
├── test_api.py           # API testing with pytest
│
├── run_id.txt            # MLflow run ID
├── best_model_uri.txt    # Model URI for loading
├── label_classes.json    # Label mapping for predictions
│
├── mlflow.db             # MLflow tracking database
├── requirements.txt
└── README_Stage4.md
```

---

## 🚀 How to Run

### 1. Train the Model

```bash
python train.py
```

This generates:

```id="t3yz3k"
run_id.txt
best_model_uri.txt
```

---

### 2. Start the API

```bash
uvicorn app:app --reload --port 5001
```

---

### 3. Access API

* Root endpoint:

```id="4q7h0h"
http://127.0.0.1:5001
```

* Interactive docs (Swagger UI):

```id="9n3t4b"
http://127.0.0.1:5001/docs
```

---

### 4. Test the API

```bash
pytest test_api.py
```

---

## API Endpoints

### GET `/`

* Returns a welcome message
* Used to verify the API is running

---

### GET `/health`

* Health check endpoint
* Returns API status

---

### POST `/predict`

* Main prediction endpoint
* Accepts JSON input and returns model predictions

#### Example Request

```json
{
  "Inventory_Reconstructed": 120,
  "Units_Sold": 45,
  "Units_Ordered": 60,
  "Price": 25.5,
  "Discount": 10,
  "Units_Sold_Lag1": 40,
  "Inventory_Change_Pct": 0.05,
  "Days_of_Stock": 10,
  "Sales_Velocity": 2.5,
  "Coverage_Ratio": 1.2,
  "Forecast_Error": 3.5,
  "Order_to_Inventory": 0.5,
  "Category": "Electronics",
  "Region": "North",
  "Weather_Condition": "Sunny",
  "Seasonality": "Summer"
}
```

#### Example Response

```json
{
  "model_run_id": "xxxxxx",
  "predictions_encoded": [1],
  "predictions_label": ["Medium Risk"]
}
```

---

## 🧠 Key Concepts

* **FastAPI**: High-performance framework for serving ML models as APIs
* **REST API**: Allows communication via HTTP requests (GET/POST)
* **Model Serving**: Making trained models accessible in production
* **MLflow Integration**: Load model artifacts using run ID / URI

---

## Notes

* Ensure `run_id.txt` and `best_model_uri.txt` exist before starting API
* API runs on port `5001`
* Input schema must match training features
* `test_api.py` validates endpoint correctness

---

## Stage 4 Completion

This stage is complete when:

* Model is successfully
