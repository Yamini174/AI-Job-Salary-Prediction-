
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("ai_job_dataset.csv")

# Basic information
print("Dataset Shape:", df.shape)
print("\nSalary Statistics:")
print(df["salary_usd"].describe())

# Salary distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["salary_usd"], bins=30, kde=True)

plt.title("Distribution of AI Job Salaries")
plt.xlabel("Salary (USD)")
plt.ylabel("Number of Jobs")

plt.tight_layout()
plt.savefig("salary_distribution.png")
plt.show()

# Salary analysis by experience level

print("\nAverage Salary by Experience Level:")
print(
    df.groupby("experience_level")["salary_usd"]
    .agg(["mean", "median", "count"])
    .round(2)
)

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="experience_level",
    y="salary_usd"
)

plt.title("Salary Distribution by Experience Level")
plt.xlabel("Experience Level")
plt.ylabel("Salary (USD)")

plt.tight_layout()
plt.savefig("salary_by_experience.png")
plt.show()

# Average salary by job title

job_salary = (
    df.groupby("job_title")["salary_usd"]
    .mean()
    .sort_values(ascending=False)
)

print("\nTop 10 Highest Paying Job Titles:")
print(job_salary.head(10).round(2))

plt.figure(figsize=(12, 7))

sns.barplot(
    x=job_salary.head(10).values,
    y=job_salary.head(10).index
)

plt.title("Top 10 Highest Paying AI Job Titles")
plt.xlabel("Average Salary (USD)")
plt.ylabel("Job Title")

plt.tight_layout()
plt.savefig("top_10_job_salaries.png")
plt.show()
