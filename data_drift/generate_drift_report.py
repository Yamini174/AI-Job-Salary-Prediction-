
import pandas as pd
from pathlib import Path

from evidently import Report
from evidently.presets import DataDriftPreset

# Project folder paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data_drift"

# Load reference and comparison datasets
reference_data = pd.read_csv(DATA_DIR / "reference_data.csv")
current_data = pd.read_csv(DATA_DIR / "comparison_data.csv")

print("Reference data shape:", reference_data.shape)
print("Current data shape:", current_data.shape)

# Create data drift report
report = Report([
    DataDriftPreset()
])

# Run drift analysis
snapshot = report.run(
    current_data=current_data,
    reference_data=reference_data
)

# Save HTML report
report_path = DATA_DIR / "data_drift_report.html"
snapshot.save_html(str(report_path))

print("\nData drift report generated successfully!")
print("Report location:", report_path)
