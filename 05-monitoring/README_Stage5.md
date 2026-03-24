Stage 5 – Monitoring
Objective

The objective of this stage is to simulate a production monitoring system that tracks model predictions, data quality, and potential drift after deployment.

This stage ensures that the deployed model remains reliable over time by introducing observability into the pipeline.

Overview

After deploying the model (Stage 4), this stage adds:

Prediction logging
Simulation of production traffic
Monitoring of prediction distributions
Data quality checks
Drift detection using Evidently

The system mimics a real-world setup where incoming data and predictions are continuously tracked and analyzed.

Architecture

Monitoring Workflow:

API receives prediction requests
Inputs and predictions are logged
Simulation script generates production-like traffic
Logs are processed and analyzed
Monitoring reports and summaries are generated
Key Components
1. Prediction Logging

The FastAPI application is extended to log:

Input data
Predictions (encoded and labeled)
Timestamp

Logs are stored in:

prediction_logs.json

Each request is recorded as a structured JSON entry

2. Traffic Simulation

A simulation script generates synthetic API requests to mimic production usage:

Randomized feature values
Multiple categorical combinations
Configurable number of requests

Command:

python simulate.py

Outputs:

predictions.csv containing inputs and predictions
3. Monitoring Pipeline

The monitoring script processes both:

Logged API data (prediction_logs.json)
Simulated data (predictions.csv)

Command:

python monitor.py
Monitoring Checks
Prediction distribution
Prediction label distribution
Missing values detection
Summary statistics for numerical features

Outputs:

monitoring_logs_flattened.csv
monitoring_summary.csv
4. Data Drift Detection

Data drift is analyzed using the Evidently library.

Command:

python evidently_report.py

Process:

Split dataset into reference and current data
Compare feature distributions
Generate drift metrics

Output:

drift_report.html
Monitoring Metrics

The system tracks:

Prediction Behavior
Class distribution (encoded and labeled)
Detection of prediction imbalance
Data Quality
Missing values
Invalid inputs
Feature Statistics
Mean, standard deviation
Range of numerical features
Data Drift
Changes in feature distributions between reference and current data
Key Implementation Details
Logging Design
Each prediction request is logged with timestamp and full input
Supports both single and batch predictions
Enables traceability of model decisions
Simulation Strategy
Random sampling ensures variability
Covers different product categories, regions, and conditions
Approximates real-world input diversity
Drift Analysis Approach
First portion of data used as reference
Remaining data treated as current production data
Statistical comparison highlights distribution shifts
Business Perspective

Monitoring is critical for maintaining model reliability in production:

Detects when model behavior changes
Identifies data quality issues early
Flags potential degradation in decision quality

Impact:

Prevents incorrect inventory decisions
Reduces operational risk
Supports continuous model improvement
Limitations
Simulation uses synthetic data, not real production traffic
Drift detection is based on simple dataset splitting
No automated alerting or retraining pipeline
Stage 5 Outcome
Prediction logging implemented
Production traffic simulated
Monitoring pipeline operational
Drift detection integrated
System provides basic observability
Next Stage

Stage 6 focuses on:

Containerization with Docker
Code quality enforcement (flake8)
CI/CD pipeline using GitHub Actions
Full pipeline automation