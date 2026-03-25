# simulate.py
# The purpose of this file is to automate calls and strengthen monitoring

# simulate.py
# The purpose of this file is to automate API calls and create predictions.csv
# for monitoring / drift analysis.

import os
import random
import time

import pandas as pd
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:5001/predict")
OUTPUT_FILE = "data/predictions.csv"

CATEGORIES = ["Electronics", "Clothing", "Home", "Groceries"]
REGIONS = ["North", "South", "East", "West"]
WEATHER = ["Sunny", "Rainy", "Cloudy"]
SEASONS = ["Summer", "Winter", "Spring", "Fall"]


def generate_random_input():
    return {
        "Inventory_Reconstructed": round(random.uniform(50, 200), 2),
        "Units_Sold": random.randint(10, 100),
        "Units_Ordered": random.randint(10, 120),
        "Price": round(random.uniform(5, 50), 2),
        "Discount": random.choice([0, 5, 10, 15, 20, 30]),
        "Units_Sold_Lag1": round(random.uniform(5, 100), 2),
        "Inventory_Change_Pct": round(random.uniform(-0.2, 0.2), 2),
        "Days_of_Stock": round(random.uniform(1, 30), 2),
        "Sales_Velocity": round(random.uniform(1, 5), 2),
        "Coverage_Ratio": round(random.uniform(0.5, 2.0), 2),
        "Forecast_Error": round(random.uniform(0, 10), 2),
        "Order_to_Inventory": round(random.uniform(0.1, 1.0), 2),
        "Category": random.choice(CATEGORIES),
        "Region": random.choice(REGIONS),
        "Weather_Condition": random.choice(WEATHER),
        "Seasonality": random.choice(SEASONS),
    }


def main(n_requests=100):
    results = []
    print("Starting simulation...")

    for i in range(n_requests):
        payload = generate_random_input()

        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()

            row = payload.copy()
            row["model_run_id"] = data.get("model_run_id")
            row["prediction_encoded"] = data.get("predictions_encoded", [None])[0]
            row["prediction_label"] = data.get("predictions_label", [None])[0]
            results.append(row)

            if (i + 1) % 20 == 0:
                print(f"Progress: {i + 1}/{n_requests}")

        except Exception as e:
            print(f"Error at iteration {i + 1}: {e}")

        time.sleep(0.05)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()