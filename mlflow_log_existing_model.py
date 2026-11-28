
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

# 1. Configure local SQLite tracking
tracking_uri = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(tracking_uri)

# 2. Create a new experiment with a local artifact folder
client = MlflowClient()

experiment_name = "AI_Job_Salary_Prediction_ML_Local"

experiment = client.get_experiment_by_name(experiment_name)

if experiment is None:
    artifact_location = Path("mlartifacts").resolve().as_uri()

    experiment_id = client.create_experiment(
        name=experiment_name,
        artifact_location=artifact_location
    )
else:
    experiment_id = experiment.experiment_id

mlflow.set_experiment(experiment_id=experiment_id)

# 3. Load the existing trained model
with open("salary_model.pkl", "rb") as file:
    model = pickle.load(file)

# 4. Load dataset and select expected features
X = pd.read_csv("X_features.csv")

feature_columns = list(model.feature_names_in_)
X = X[feature_columns]

# 5. Prepare model signature
input_example = X.head(5)
predictions = model.predict(input_example)

signature = infer_signature(input_example, predictions)

# 6. Log the existing model
with mlflow.start_run(run_name="Existing_Trained_Model_Logging"):

    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("pipeline_type", "Scikit-learn Pipeline")
    mlflow.log_param("preprocessing", "ColumnTransformer")
    mlflow.log_param("input_features", len(feature_columns))
    mlflow.log_param("available_dataset_records", len(X))

    mlflow.sklearn.log_model(
        sk_model=model,
        name="salary_prediction_model",
        signature=signature,
        input_example=input_example,
        skops_trusted_types=[
            "sklearn.compose._column_transformer._RemainderColsList",
            "sklearn.tree._tree.Tree"
        ]
    )

    print("Existing model logged successfully!")
    print("Model: RandomForestRegressor")
    print("Features:", len(feature_columns))
    print("Dataset records:", len(X))

print("MLflow logging completed!")
