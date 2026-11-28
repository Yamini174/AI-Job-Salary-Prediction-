
import mlflow
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPORT_PATH = PROJECT_DIR / "data_drift" / "data_drift_report.html"

if not REPORT_PATH.exists():
    raise FileNotFoundError(f"Report not found: {REPORT_PATH}")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("AI_Job_Salary_Prediction_ML_Local")

with mlflow.start_run(run_name="Data_Drift_Report_Logging"):

    mlflow.log_param("monitoring_tool", "Evidently AI")
    mlflow.log_param("evidently_version", "0.7.23")
    mlflow.log_param("reference_records", 10500)
    mlflow.log_param("comparison_records", 4500)
    mlflow.log_param("total_features", 18)
    mlflow.log_param("drift_threshold", 0.5)

    mlflow.log_metric("drifted_features", 0)
    mlflow.log_metric("drifted_feature_share", 0.0)

    mlflow.log_artifact(
        str(REPORT_PATH),
        artifact_path="data_drift"
    )

    print("Data drift report logged successfully!")

print("MLflow data drift logging completed!")
