Stage 2 – Feature Engineering & Data Understanding
Objective

The objective of this stage is to transform raw inventory data into a structured feature set suitable for modeling, while developing a clear understanding of how business assumptions affect risk classification.

This stage focuses on:

Feature engineering
Target label construction
Scenario-based analysis
Sensitivity analysis of thresholds
Overview

Building on the baseline pipeline, this stage introduces:

Derived features capturing inventory dynamics
A formalized labeling strategy for risk prediction
Multiple scenarios reflecting different business assumptions
Analysis of how threshold choices influence class distribution

This stage is analytical in nature and does not involve model training or optimization.

Dataset
Property	Value
Source	Retail inventory dataset (Kaggle)
Granularity	Daily per Store × Product
Target	Inventory Risk (multiclass)
Feature Engineering

New features are created to better capture operational and demand patterns:

Inventory Dynamics
Inventory_Change_Pct
Days_of_Stock
Demand & Sales Behavior
Sales_Velocity
Units_Sold_Lag1
Forecast Reliability
Forecast_Error
Operational Ratios
Coverage_Ratio
Order_to_Inventory

These features aim to represent:

Short-term demand trends
Inventory sustainability
Forecast uncertainty
Supply-demand balance
Target Label Definition

The target variable (Risk_Label) is constructed using threshold-based rules derived from business logic.

Classes
Low Risk
Medium Risk
High Risk
Core Logic
High Risk (Stockout)
Inventory is insufficient relative to expected demand
Medium Risk (Overstock)
Inventory is excessive and sales velocity is low
Low Risk (Safe Zone)
Inventory is balanced relative to demand
Scenario Design

Three scenarios are defined to simulate different business perspectives:

Scenario 1 – Balanced
Moderate thresholds
Balanced trade-off between false positives and false negatives
Scenario 2 – Conservative
Stricter thresholds
Fewer risk flags
Higher chance of missing true risks
Scenario 3 – Sensitive
Looser thresholds
More aggressive risk detection
Higher false positive rate
Sensitivity Analysis (Section 2.6)

A sensitivity analysis is conducted to evaluate how threshold choices affect:

Class distribution
Risk detection behavior
Business implications
Key Insight
Lower thresholds → more items flagged as risk
Higher thresholds → fewer risk alerts but increased missed risks
Important Distinction

This analysis is:

Not model tuning
Not optimization

It is strictly:

Exploration of how business assumptions shape the target variable
Business Interpretation

Each scenario reflects a different operational strategy:

Conservative (Scenario 2):
Minimizes unnecessary interventions but risks stockouts
Sensitive (Scenario 3):
Maximizes risk detection but may trigger excessive actions
Balanced (Scenario 1):
Provides a compromise between the two extremes

Key conclusion:

The “best” scenario depends on business cost trade-offs, not statistical metrics.

Data Preparation for Modeling

Final outputs of this stage include:

Cleaned and feature-engineered dataset
Defined target variable (Risk_Label)
Structured feature set for modeling
Scenario-based labeled datasets

These outputs are used directly in Stage 3 for model training and evaluation.

Key Outcomes
Robust feature set capturing inventory dynamics
Clear and interpretable labeling strategy
Multiple business-driven scenarios
Understanding of threshold sensitivity
Limitations
Labels are derived from heuristic rules, not ground truth
Threshold selection introduces subjectivity
Dataset remains synthetic, which may inflate separability
Stage 2 Outcome
Feature engineering pipeline completed
Target variable defined and validated
Scenario analysis performed
Dataset ready for modeling in Stage 3
Next Stage

Stage 3 focuses on:

Training multiple machine learning models
Evaluating performance using validation metrics
Tracking experiments using MLflow
Selecting the best-performing model