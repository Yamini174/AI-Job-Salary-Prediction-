
import pandas as pd

df = pd.read_csv("ai_job_salary_engineered.csv")

X = df.drop(columns=["salary_usd", "job_id"])

# Identify numerical columns
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

# Identify categorical columns
categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("Numerical Features:")
print(numerical_features)

print("\nTotal Numerical Features:", len(numerical_features))

print("\nCategorical Features:")
print(categorical_features)

print("\nTotal Categorical Features:", len(categorical_features))

print("\nTotal Input Features:", X.shape[1])
