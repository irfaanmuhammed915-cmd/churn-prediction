"""
Synthetic telecom churn dataset generator.
Produces a realistic dataset with ~30% churn rate (industry-typical).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 7043  # same size as IBM Telco churn dataset


def generate():
    gender = np.random.choice(["Male", "Female"], N)
    senior = np.random.choice([0, 1], N, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], N, p=[0.48, 0.52])
    dependents = np.where(partner == "Yes",
                          np.random.choice(["Yes", "No"], N, p=[0.36, 0.64]),
                          np.random.choice(["Yes", "No"], N, p=[0.17, 0.83]))
    tenure = np.random.exponential(scale=32, size=N).clip(1, 72).astype(int)
    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.21, 0.24])
    paperless = np.random.choice(["Yes", "No"], N, p=[0.59, 0.41])
    payment = np.random.choice(
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
        N, p=[0.34, 0.23, 0.22, 0.21])
    phone = np.random.choice(["Yes", "No"], N, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone == "Yes",
        np.random.choice(["Yes", "No"], N, p=[0.42, 0.58]),
        "No phone service")
    internet = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22])
    online_security = np.where(
        internet != "No", np.random.choice(["Yes", "No"], N, p=[0.29, 0.71]), "No internet service")
    online_backup = np.where(
        internet != "No", np.random.choice(["Yes", "No"], N, p=[0.34, 0.66]), "No internet service")
    device_protection = np.where(
        internet != "No", np.random.choice(["Yes", "No"], N, p=[0.34, 0.66]), "No internet service")
    tech_support = np.where(
        internet != "No", np.random.choice(["Yes", "No"], N, p=[0.29, 0.71]), "No internet service")
    streaming_tv = np.where(
        internet != "No", np.random.choice(["Yes", "No"], N, p=[0.38, 0.62]), "No internet service")
    streaming_movies = np.where(
        internet != "No", np.random.choice(["Yes", "No"], N, p=[0.39, 0.61]), "No internet service")
    monthly_charges = np.where(
        internet == "No", np.random.normal(21, 3, N),
        np.where(internet == "DSL", np.random.normal(59, 15, N),
                 np.random.normal(76, 18, N))).clip(18, 120)
    total_charges = (tenure * monthly_charges * np.random.uniform(0.95, 1.05, N)).round(2)

    log_odds = (
        -4.5
        + 0.04 * (72 - tenure)
        + 1.8 * (contract == "Month-to-month").astype(float)
        - 0.7 * (contract == "Two year").astype(float)
        + 0.9 * (internet == "Fiber optic").astype(float)
        - 0.5 * (online_security == "Yes").astype(float)
        - 0.4 * (tech_support == "Yes").astype(float)
        + 0.6 * (payment == "Electronic check").astype(float)
        + 0.4 * senior.astype(float)
        + 0.02 * (monthly_charges - 50)
        + np.random.normal(0, 0.5, N)
    )
    churn_prob = 1 / (1 + np.exp(-log_odds))
    churn = (np.random.uniform(size=N) < churn_prob).astype(int)

    return pd.DataFrame({
        "customerID": [f"CUST-{i:05d}" for i in range(N)],
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges.round(2),
        "TotalCharges": total_charges,
        "Churn": churn,
    })


if __name__ == "__main__":
    df = generate()
    df.to_csv("telco_churn.csv", index=False)
    print(f"Dataset saved: {len(df)} rows, {df['Churn'].mean():.1%} churn rate")
