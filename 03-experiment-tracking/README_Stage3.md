# Stage 3: Experiment Tracking with MLflow

## Objective

The goal of this stage is to introduce experiment tracking into the machine learning workflow using MLflow. This allows us to systematically compare different models, monitor their performance, and select the best model based on objective metrics.

---

## Overview

After building models in Stage 2, we integrate MLflow to:

* Track model parameters and configurations
* Log performance metrics across different datasets
* Store evaluation artifacts for analysis
* Compare multiple experiments efficiently

This ensures that model selection is reproducible and transparent.

---

## MLflow Setup

MLflow is configured to run locally using a SQLite database:

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("inventory-risk-stage3")
```

All runs, metrics, and artifacts are stored locally and can be accessed through the MLflow UI.

---

## Models Evaluated

We trained and compared the following models:

* Logistic Regression
* Random Forest
* XGBoost

Each model is implemented as part of a pipeline to ensure consistent preprocessing.

---

## Logged Information

For each experiment run, the following were recorded:

### Parameters

* Model type
* Feature sets (numerical and categorical)
* Number of target classes

### Metrics

* Accuracy
* Precision (macro)
* Recall (macro)
* F1-score (macro)

Metrics were logged for:

* Training set
* Validation set
* Test set

---

## Artifacts

In addition to metrics, we logged:

* Confusion matrices (CSV and image)
* Classification reports (text format)
* Trained model objects using MLflow

These artifacts help provide deeper insights into model performance.

---

## Model Selection

Models were compared based on validation performance using the following priority:

1. F1-score (macro)
2. Recall (macro)
3. Precision (macro)
4. Accuracy

The best-performing model was selected automatically and its details were saved.

---

## Outputs

The following files were generated:

* `results_df.csv` → summary of all model performances
* `run_id.txt` → MLflow run ID of the selected model
* `best_model_uri.txt` → path to the best model in MLflow
* `label_classes.json` → mapping of encoded labels to class names

---

## Experiment Scenarios

Two MLflow configurations were considered:

### Scenario 1: Local Tracking

MLflow runs locally using a SQLite database. All experiment data is stored on a single machine.

* Suitable for individual development
* Easy to set up
* Limited collaboration

---

### Scenario 2: Centralized Tracking

MLflow connects to a shared tracking server, allowing multiple users to log and compare experiments.

* Enables team collaboration
* Centralized experiment history
* More scalable setup

---

### Key Difference

| Aspect        | Scenario 1 (Local) | Scenario 2 (Centralized) |
| ------------- | ------------------ | ------------------------ |
| Storage       | Local machine      | Remote server            |
| Users         | Single user        | Multiple users           |
| Collaboration | Limited            | High                     |
| Setup         | Simple             | More advanced            |

---

## MLflow UI

To view experiment results:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

This interface allows you to:

* Compare runs
* Visualize metrics
* Inspect artifacts

---

## Conclusion

Stage 3 enhances the pipeline by introducing structured experiment tracking. MLflow enables reproducibility, simplifies model comparison, and ensures that the best model is selected using consistent and transparent criteria.

---
