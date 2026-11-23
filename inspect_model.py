
import pickle

# Load the trained model
with open("salary_model.pkl", "rb") as file:
    model = pickle.load(file)

print("MODEL LOADED SUCCESSFULLY!")
print("Model type:", type(model))

if hasattr(model, "feature_names_in_"):
    print("\nExpected features:")
    print(list(model.feature_names_in_))

if hasattr(model, "n_features_in_"):
    print("\nNumber of features:", model.n_features_in_)
