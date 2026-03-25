# app.py
# Stage 6 deployment version:
# load local model artifacts only (no MLflow)
# define FastAPI app
# expose /, /health, /predict

import os
import json
import pickle
from typing import List, Union

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
LABEL_PATH = os.path.join(BASE_DIR, "models", "label_classes.json")

FEATURE_COLUMNS = [
    "Inventory_Reconstructed",
    "Units Sold",
    "Units Ordered",
    "Price",
    "Discount",
    "Units_Sold_Lag1",
    "Inventory_Change_Pct",
    "Days_of_Stock",
    "Sales_Velocity",
    "Coverage_Ratio",
    "Forecast_Error",
    "Order_to_Inventory",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality",
]

INTEGER_FEATURES = [
    "Units Sold",
    "Units Ordered",
    "Discount",
]

FLOAT_FEATURES = [
    "Inventory_Reconstructed",
    "Price",
    "Units_Sold_Lag1",
    "Inventory_Change_Pct",
    "Days_of_Stock",
    "Sales_Velocity",
    "Coverage_Ratio",
    "Forecast_Error",
    "Order_to_Inventory",
]

CATEGORICAL_FEATURES = [
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality",
]


class PredictionInput(BaseModel):
    Inventory_Reconstructed: float
    Units_Sold: int
    Units_Ordered: int
    Price: float
    Discount: int
    Units_Sold_Lag1: float
    Inventory_Change_Pct: float
    Days_of_Stock: float
    Sales_Velocity: float
    Coverage_Ratio: float
    Forecast_Error: float
    Order_to_Inventory: float
    Category: str
    Region: str
    Weather_Condition: str
    Seasonality: str

    def to_model_dict(self) -> dict:
        return {
            "Inventory_Reconstructed": self.Inventory_Reconstructed,
            "Units Sold": self.Units_Sold,
            "Units Ordered": self.Units_Ordered,
            "Price": self.Price,
            "Discount": self.Discount,
            "Units_Sold_Lag1": self.Units_Sold_Lag1,
            "Inventory_Change_Pct": self.Inventory_Change_Pct,
            "Days_of_Stock": self.Days_of_Stock,
            "Sales_Velocity": self.Sales_Velocity,
            "Coverage_Ratio": self.Coverage_Ratio,
            "Forecast_Error": self.Forecast_Error,
            "Order_to_Inventory": self.Order_to_Inventory,
            "Category": self.Category,
            "Region": self.Region,
            "Weather Condition": self.Weather_Condition,
            "Seasonality": self.Seasonality,
        }


def load_labels() -> List[str]:
    if not os.path.exists(LABEL_PATH):
        raise FileNotFoundError(f"label_classes.json not found at {LABEL_PATH}")

    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "class_names" in data:
        return data["class_names"]

    if isinstance(data, list):
        return data

    raise ValueError("label_classes.json format is invalid.")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"model.pkl not found at {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def prepare_input(payload: Union[dict, List[dict]]) -> pd.DataFrame:
    if isinstance(payload, dict):
        df = pd.DataFrame([payload])
    elif isinstance(payload, list):
        df = pd.DataFrame(payload)
    else:
        raise ValueError("Input must be a dictionary or list of dictionaries.")

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    df = df[FEATURE_COLUMNS].copy()

    for col in INTEGER_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in FLOAT_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_int_cols = df[INTEGER_FEATURES].columns[df[INTEGER_FEATURES].isnull().any()].tolist()
    bad_float_cols = df[FLOAT_FEATURES].columns[df[FLOAT_FEATURES].isnull().any()].tolist()

    if bad_int_cols or bad_float_cols:
        raise ValueError(
            "These numerical columns contain invalid or missing values: "
            f"{bad_int_cols + bad_float_cols}"
        )

    for col in INTEGER_FEATURES:
        df[col] = df[col].astype("int64")

    for col in FLOAT_FEATURES:
        df[col] = df[col].astype("float64")

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("object")

    return df


LABELS = load_labels()
model = load_model()

app = FastAPI(title="Inventory Risk API", version="1.0")


@app.get("/")
def home():
    return {
        "message": "Inventory risk model is running.",
        "endpoint": "/predict",
        "required_features": [
            "Inventory_Reconstructed",
            "Units_Sold",
            "Units_Ordered",
            "Price",
            "Discount",
            "Units_Sold_Lag1",
            "Inventory_Change_Pct",
            "Days_of_Stock",
            "Sales_Velocity",
            "Coverage_Ratio",
            "Forecast_Error",
            "Order_to_Inventory",
            "Category",
            "Region",
            "Weather_Condition",
            "Seasonality",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: Union[PredictionInput, List[PredictionInput]]):
    try:
        if isinstance(payload, list):
            records = [item.to_model_dict() for item in payload]
        else:
            records = payload.to_model_dict()

        X = prepare_input(records)
        preds = model.predict(X)

        predictions_encoded = []
        predictions_label = []

        for pred in preds:
            try:
                pred_int = int(pred)
                predictions_encoded.append(pred_int)

                if 0 <= pred_int < len(LABELS):
                    predictions_label.append(LABELS[pred_int])
                else:
                    predictions_label.append(str(pred))
            except Exception:
                predictions_encoded.append(str(pred))
                predictions_label.append(str(pred))

        return {
            "predictions_encoded": predictions_encoded,
            "predictions_label": predictions_label,
        }

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc