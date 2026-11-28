
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data_drift"

# Load engineered dataset
df = pd.read_csv(PROJECT_DIR / "ai_job_salary_engineered.csv")

print("Original dataset shape:", df.shape)

# Use the model's 18 expected input features
features = [
    "job_title",
    "salary_currency",
    "experience_level",
    "employment_type",
    "company_location",
    "company_size",
    "employee_residence",
    "remote_ratio",
    "education_required",
    "years_experience",
    "industry",
    "job_description_length",
    "benefits_score",
    "company_name",
    "posting_year",
    "posting_month",
    "days_to_deadline",
    "skills_count"
]

# Select only model input features
X = df[features].copy()

# Split into reference and comparison samples
reference_data = X.sample(frac=0.70, random_state=42)
comparison_data = X.drop(reference_data.index)

# Save separate datasets
reference_data.to_csv(DATA_DIR / "reference_data.csv", index=False)
comparison_data.to_csv(DATA_DIR / "comparison_data.csv", index=False)

print("\nData drift datasets created successfully!")
print("Reference data shape:", reference_data.shape)
print("Comparison data shape:", comparison_data.shape)
print("\nReference file:", DATA_DIR / "reference_data.csv")
print("Comparison file:", DATA_DIR / "comparison_data.csv")
