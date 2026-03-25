# Stage 5: Monitoring & Drift Detection

## 📌 Overview

In this stage, we extend our deployed ML API by adding monitoring capabilities.
We simulate real-world predictions, log outputs, and analyze potential data drift and model performance using both custom scripts and Evidently AI.

---

## ⚙️ Pipeline Flow

The monitoring pipeline follows this structure:

```
FastAPI (app.py)
        ↓
simulate.py → generates predictions
        ↓
data/predictions.csv
        ↓
monitor.py → basic summary report
        ↓
evidently_report.py → advanced drift analysis
```

---

## 📁 Project Structure

```
05-monitoring/
│
├── app.py                  # FastAPI service (same as Stage 4)
├── simulate.py             # Simulates API calls
├── monitor.py              # Basic monitoring summary
├── evidently_report.py     # Drift detection with Evidently
├── test_api.py             # API tests
│
├── data/
│   └── predictions.csv     # Simulated predictions
│
├── monitoring_report.html  # Basic report
├── evidently_report.html   # Drift dashboard
│
├── run_id.txt
├── best_model_uri.txt
├── label_classes.json
├── requirements.txt
└── README_Stage5.md
```

---

## 🚀 How to Run

### 1. Start API

```bash
uvicorn app:app --reload --port 5001
```

### 2. Test API

```bash
pytest test_api.py
```

### 3. Simulate Predictions

```bash
python simulate.py
```

This generates:

```
data/predictions.csv
```

---

### 4. Run Basic Monitoring

```bash
python monitor.py
```

Generates:

```
monitoring_report.html
```

---

### 5. Run Evidently Report

```bash
python evidently_report.py
```

Generates:

```
evidently_report.html
```

---

## 📊 Monitoring Outputs

### 🔹 Basic Monitoring (`monitor.py`)

* Prediction distribution
* Label distribution
* Missing values
* Numerical feature summary

---

### 🔹 Advanced Monitoring (Evidently)

* Data drift detection
* Feature distribution comparison
* Reference vs current data analysis

---

## 🧠 Key Concepts

* **Simulation**: Mimics real-time API usage
* **Monitoring**: Tracks model behavior post-deployment
* **Data Drift**: Detects changes in input data distribution
* **Reference vs Current Split**: First half vs second half of data

---

## ⚠️ Notes

* API must be running before executing `simulate.py`
* Predictions are stored in `data/predictions.csv`
* Evidently compares earlier vs later predictions to detect drift

---

## ✅ Stage 5 Completion

This stage is complete when:

* API is functional
* Simulation generates predictions
* Monitoring reports are created
* Evidently report visualizes drift

---

## 🔜 Next Step

Proceed to **Stage 6: Automation & CI/CD**, where the full pipeline will be automated and deployed.

---
