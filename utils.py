import pandas as pd
def clean_total_charges(X):
    X = X.copy()
    X["TotalCharges"] = X["TotalCharges"].astype(str).str.strip()
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
    X["TotalCharges"] = X["TotalCharges"].fillna(X["TotalCharges"].median())
    return X