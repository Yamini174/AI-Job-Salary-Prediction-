
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


def test_model_file_exists():
    assert Path("salary_model.pkl").exists()


def test_model_loads_successfully():
    model = joblib.load("salary_model.pkl")
    assert model is not None


def test_model_has_predict_method():
    model = joblib.load("salary_model.pkl")
    assert hasattr(model, "predict")


def test_model_has_expected_features():
    model = joblib.load("salary_model.pkl")

    expected_features = [
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

    assert list(model.feature_names_in_) == expected_features


def test_model_can_make_prediction():
    model = joblib.load("salary_model.pkl")

    sample_data = {
        "job_title": "AI Research Scientist",
        "salary_currency": "USD",
        "experience_level": "SE",
        "employment_type": "CT",
        "company_location": "China",
        "company_size": "M",
        "employee_residence": "China",
        "remote_ratio": 50,
        "education_required": "Bachelor",
        "years_experience": 9,
        "industry": "Automotive",
        "job_description_length": 1076,
        "benefits_score": 5.9,
        "company_name": "Smart Analytics",
        "posting_year": 2024,
        "posting_month": 10,
        "days_to_deadline": 20,
        "skills_count": 5
    }

    sample = pd.DataFrame([sample_data])
    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert np.isfinite(prediction[0])
