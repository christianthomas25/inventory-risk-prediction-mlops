# Stage 2: Feature Engineering, Scenario Analysis & Baseline Modeling

## Overview

Stage 2 transforms raw data into a structured, model-ready dataset and validates the pipeline through baseline models.

This stage bridges data preparation and modeling by integrating business logic into the ML workflow:

**Feature Engineering → Scenario Design → Target Construction → Temporal Split → Baseline Modeling**

---

## 1. Feature Engineering

Key transformations are applied to capture inventory dynamics and demand behavior:

### Derived Features

* **Lag Features**

  * `Units_Sold_Lag1`
* **Rolling / Trend Features**

  * Inventory change and trends
* **Inventory & Demand Ratios**

  * `Days_of_Stock`
  * `Sales_Velocity`
  * `Coverage_Ratio`
  * `Order_to_Inventory`
* **Forecast-Based Features**

  * `Forecast_Error`
* **Reconstructed Inventory**

  * `Inventory_Reconstructed` (core variable used consistently across labeling and modeling)

Categorical variables (e.g., Category, Region, Seasonality) are preserved for encoding.

---

## 2. Scenario Analysis (Business Logic)

Three threshold configurations are evaluated to define inventory risk behavior.

### Scenario 1: Conservative Setting

Allows a wider inventory buffer before labeling overstock.
Reduces false overstock alerts but tolerates higher inventory levels to avoid stockouts.

### Scenario 2: Stricter / More Sensitive Setting

Detects stock imbalances earlier.
Reflects tighter inventory control with faster reaction to demand mismatches.

### Scenario 3: Balanced Setting

Compromise between conservative and strict approaches.
Balances stockout prevention with timely overstock detection.

Scenario comparison is performed using class distribution analysis to select a suitable configuration.

---

## 3. Target Construction

Two labels are created:

* **Risk_Label_Current** → current-period classification
* **Risk_Label** → next-period prediction target

```python
df["Risk_Label"] = df.groupby(["Store ID", "Product ID"])["Risk_Label_Current"].shift(-1)
```

This ensures:

* Alignment with real-world decision timing
* No data leakage from future information

Rows with missing future labels are removed.

---

## 4. Temporal Train / Validation / Test Split

Data is split chronologically:

* **Train:** before first cutoff
* **Validation:** between cutoffs
* **Test:** after second cutoff

This simulates real-world deployment and prevents information leakage across time.

---

## 5. Data Preparation

* Missing values are imputed using **training-set medians**
* Leakage-prone columns are removed
* Target variables are excluded from feature inputs
* Categorical variables are **one-hot encoded**
* Numerical features are scaled where appropriate

---

## 6. Baseline Modeling

Three baseline models are implemented:

* **Logistic Regression** (interpretable baseline)
* **Random Forest** (non-linear benchmark)
* **XGBoost** (boosting-based model)

### Class Imbalance Handling

```python
class_weight="balanced"
```

### Evaluation Metrics

* Accuracy
* Precision / Recall / F1-score
* Confusion Matrix

Label encoding is aligned with model outputs to ensure correct interpretation of results.

---

## 7. Key Design Decisions

* **Inventory consistency:** `Inventory_Reconstructed` used across features, labeling, and modeling
* **Temporal split:** prevents future data leakage
* **Scenario-driven labeling:** integrates business logic into ML pipeline
* **Proper label encoding:** ensures correct evaluation metrics

---

## 8. Outputs

* Cleaned datasets: **train / validation / test**
* Feature matrices ready for modeling
* Scenario comparison results
* Baseline model performance metrics

---

## Summary

Stage 2 establishes a robust and production-aligned ML pipeline by combining:

* Business-driven labeling
* Feature engineering
* Proper evaluation methodology
* Baseline model validation

This stage prepares the foundation for:

* Experiment tracking (Stage 3)
* Model optimization
* Deployment and monitoring
