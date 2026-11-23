
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="AI Job Salary Prediction",
    page_icon="💼",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    with open("salary_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

st.title("💼 AI Job Salary Prediction")
st.write("Predict job salaries using Machine Learning.")

st.header("Enter Job Details")

job_title = st.text_input("Job Title", "Data Analyst")
salary_currency = st.text_input("Salary Currency", "USD")

experience_level = st.selectbox(
    "Experience Level",
    ["Entry-level", "Mid-level", "Senior", "Executive"]
)

employment_type = st.selectbox(
    "Employment Type",
    ["Full-time", "Part-time", "Contract", "Freelance"]
)

company_location = st.text_input("Company Location", "United States")
company_size = st.selectbox("Company Size", ["S", "M", "L"])
employee_residence = st.text_input("Employee Residence", "United States")

remote_ratio = st.number_input("Remote Ratio", 0, 100, 100)
education_required = st.text_input("Education Required", "Bachelor's")
years_experience = st.number_input("Years of Experience", 0, 50, 1)
industry = st.text_input("Industry", "Technology")

job_description_length = st.number_input(
    "Job Description Length", 0, 10000, 500
)

benefits_score = st.number_input("Benefits Score", 0.0, 10.0, 5.0)
company_name = st.text_input("Company Name", "Example Company")

posting_year = st.number_input("Posting Year", 2020, 2035, 2026)
posting_month = st.number_input("Posting Month", 1, 12, 11)
days_to_deadline = st.number_input("Days to Deadline", 0, 365, 30)
skills_count = st.number_input("Number of Required Skills", 0, 100, 5)

if st.button("Predict Salary"):
    input_data = pd.DataFrame([{
        "job_title": job_title,
        "salary_currency": salary_currency,
        "experience_level": experience_level,
        "employment_type": employment_type,
        "company_location": company_location,
        "company_size": company_size,
        "employee_residence": employee_residence,
        "remote_ratio": remote_ratio,
        "education_required": education_required,
        "years_experience": years_experience,
        "industry": industry,
        "job_description_length": job_description_length,
        "benefits_score": benefits_score,
        "company_name": company_name,
        "posting_year": posting_year,
        "posting_month": posting_month,
        "days_to_deadline": days_to_deadline,
        "skills_count": skills_count
    }])

    prediction = model.predict(input_data)

    st.success(f"Predicted Salary: ${prediction[0]:,.2f} USD")
