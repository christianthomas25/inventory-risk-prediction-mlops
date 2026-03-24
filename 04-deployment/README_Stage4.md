Stage 4 – Model Deployment (FastAPI)
Objective

The objective of this stage is to deploy the trained inventory risk prediction model as a REST API using FastAPI. This enables real-time inference and simulates how the model would be used in a production environment.

Overview

In this stage, the best-performing model from Stage 3 (tracked via MLflow) is:

Loaded dynamically
Integrated into a FastAPI application
Exposed through HTTP endpoints for prediction and system health monitoring

The API supports:

Single and batch predictions
Input validation
Error handling
Architecture

Workflow:

Load model (MLflow or packaged model)
Receive request via /predict
Validate input using Pydantic
Transform input into model-compatible format
Generate predictions
Return encoded and human-readable outputs
Key Files
app.py – FastAPI application and model loading logic
test_api.py – API testing for health and prediction endpoints
run_id.txt – Stores MLflow run ID of the best model
best_model_uri.txt – Stores model URI
packaged_model/ – Portable model used for deployment
Running the API
1. Install dependencies
pip install -r requirements.txt
2. Start the API
uvicorn app:app --reload --port 8000
API Endpoints
Root Endpoint
GET /

Returns API metadata and required input features.

Health Check
GET /health

Response:

{
  "status": "ok"
}
Prediction Endpoint
POST /predict
Example Input
{
  "Inventory_Level": 120,
  "Units_Sold": 35,
  "Units_Ordered": 40,
  "Price": 19.99,
  "Discount": 0,
  "Units_Sold_Lag1": 30,
  "Inventory_Change_Pct": 0.08,
  "Days_of_Stock": 12,
  "Sales_Velocity": 2.9,
  "Coverage_Ratio": 1.4,
  "Forecast_Error": 3.5,
  "Order_to_Inventory": 0.33,
  "Category": "Electronics",
  "Region": "North",
  "Weather_Condition": "Sunny",
  "Seasonality": "Summer"
}
Example Output
{
  "predictions_encoded": [1],
  "predictions_label": ["Medium Risk"]
}
Input Validation

Input validation is handled using a Pydantic schema:

Ensures correct data types
Converts API field names to model feature names
Detects missing or invalid features

Feature types:

Integers: inventory, units sold, discount
Floats: price, ratios, derived metrics
Categorical: category, region, weather, seasonality
Model Loading Strategy

The application supports multiple model loading options, in the following priority:

packaged_model/ (preferred for portability)
MODEL_URI environment variable
best_model_uri.txt
run_id.txt

This design ensures flexibility during development and portability in containerized environments.

Testing

Run tests using:

pytest test_api.py

Test coverage includes:

/health endpoint returns correct status
/predict endpoint returns valid structure and predictions
Key Implementation Details
Feature Name Mapping

API input format:

Inventory_Level

Model training format:

Inventory Level

This mismatch is handled through a transformation layer in the input schema.

Data Type Consistency

The system enforces strict data typing:

Numerical values are cast to appropriate types
Invalid or missing values trigger explicit errors
Label Decoding

Model outputs numerical classes which are mapped to:

Low Risk
Medium Risk
High Risk
Business Perspective

This stage converts the trained model into a usable decision system:

Enables real-time inventory risk assessment
Can be integrated into operational systems such as dashboards or ERP platforms
Supports automated decision-making processes

Impact:

Improved inventory control
Reduced stockouts and overstocking
Scalable prediction service
Stage 4 Outcome
Model successfully deployed as a REST API
Endpoints implemented and tested
Input validation and error handling in place
Portable model loading strategy established
System ready for monitoring in Stage 5