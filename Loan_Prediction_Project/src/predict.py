import joblib
import pandas as pd

# Load trained model and scaler
model = joblib.load("models/xgboost.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load dataset just to get correct feature order
df = pd.read_csv("data/Loan_default.csv")
df.drop(columns=['LoanID','Loan_ID','Id'], inplace=True, errors='ignore')

# Get feature names used in training
features = df.drop('Default', axis=1).columns

# Example input (replace values if needed)
sample_values = [45,500000,200000,720,36,5,3,12.5,24,300000,1,1,0,1,0,0]

# Create dataframe with correct feature names
sample_df = pd.DataFrame([sample_values], columns=features)

# Scale input
sample_scaled = scaler.transform(sample_df)

# Predict
prediction = model.predict(sample_scaled)
probabilities = model.predict_proba(sample_scaled)[0]

if prediction[0] == 1:
    print("Prediction: High Risk (Loan Default Likely)")
else:
    print("Prediction: Low Risk (Loan Repayment Likely)")
