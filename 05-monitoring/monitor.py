# monitor.py
# Reads predictions.csv and generates a monitoring report

import os
import pandas as pd

INPUT_FILE = "data/predictions.csv"
OUTPUT_HTML = "monitoring_report.html"


def main():
    if not os.path.exists(INPUT_FILE):
        print("No predictions file found.")
        return

    df = pd.read_csv(INPUT_FILE)

    print("\n=== Monitoring Summary ===")
    print(f"Rows: {len(df)}")

    print("\n=== Prediction Distribution ===")
    print(df["prediction_encoded"].value_counts())

    print("\n=== Prediction Label Distribution ===")
    print(df["prediction_label"].value_counts())

    print("\n=== Missing Values ===")
    print(df.isnull().sum())

    numeric_cols = [
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
    ]

    available_cols = [c for c in numeric_cols if c in df.columns]

    summary = df[available_cols].describe().T

    # Save HTML report
    with open(OUTPUT_HTML, "w") as f:
        f.write("<h1>Monitoring Report</h1>")
        f.write("<h2>Prediction Distribution</h2>")
        f.write(df["prediction_encoded"].value_counts().to_frame().to_html())

        f.write("<h2>Prediction Labels</h2>")
        f.write(df["prediction_label"].value_counts().to_frame().to_html())

        f.write("<h2>Missing Values</h2>")
        f.write(df.isnull().sum().to_frame().to_html())

        f.write("<h2>Numerical Summary</h2>")
        f.write(summary.to_html())

    print("\nReport saved as monitoring_report.html")


if __name__ == "__main__":
    main()