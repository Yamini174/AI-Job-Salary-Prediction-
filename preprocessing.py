
import pandas as pd

# Load engineered dataset
df = pd.read_csv("ai_job_salary_engineered.csv")

# Separate target variable
X = df.drop(columns=["salary_usd", "job_id"])

# Target variable
y = df["salary_usd"]

# Display results
print("Input Features Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nInput Columns:")
print(X.columns.tolist())

print("\nTarget Variable:")
print(y.head())

# Save input features and target
X.to_csv("X_features.csv", index=False)
y.to_csv("y_target.csv", index=False)

print("\nFeature and target separation completed!")
