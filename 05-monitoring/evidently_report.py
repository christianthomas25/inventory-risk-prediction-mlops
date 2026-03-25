# evidently_report.py

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

df = pd.read_csv("data/predictions.csv")

mid = len(df) // 2
reference_data = df.iloc[:mid].copy()
current_data = df.iloc[mid:].copy()

report = Report([
    DataDriftPreset()
])

my_eval = report.run(current_data=current_data, reference_data=reference_data)

my_eval.save_html("evidently_report.html")

print("Evidently report generated: evidently_report.html")