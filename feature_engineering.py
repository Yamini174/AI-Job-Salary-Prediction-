
import pandas as pd

# Load original dataset
df = pd.read_csv("ai_job_dataset.csv")

# Convert date columns into datetime format
df["posting_date"] = pd.to_datetime(df["posting_date"])
df["application_deadline"] = pd.to_datetime(
    df["application_deadline"]
)

# Extract posting year and month
df["posting_year"] = df["posting_date"].dt.year
df["posting_month"] = df["posting_date"].dt.month

# Calculate days between posting and deadline
df["days_to_deadline"] = (
    df["application_deadline"] - df["posting_date"]
).dt.days

# Count required skills
df["skills_count"] = (
    df["required_skills"]
    .fillna("")
    .apply(lambda x: len([s for s in x.split(",") if s.strip()]))
)

# Display new features
print("New Feature Values:")
print(df[
    [
        "posting_year",
        "posting_month",
        "days_to_deadline",
        "skills_count"
    ]
].head())

print("\nUpdated Dataset Shape:", df.shape)

# Save engineered dataset
df.to_csv("ai_job_salary_engineered.csv", index=False)

print("\nFeature engineering completed successfully!")
