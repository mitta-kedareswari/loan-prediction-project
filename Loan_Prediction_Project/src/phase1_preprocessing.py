# PHASE 1: Data Cleaning & Preprocessing (Memory Safe)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv("../data/Loan_default.csv")
print("Dataset loaded successfully")
print(df.head())

# -------------------------------
# 2. Drop unnecessary ID columns
# -------------------------------
df.drop(columns=['LoanID', 'Loan_ID', 'Id'], inplace=True, errors='ignore')

# -------------------------------
# 3. Check missing values
# -------------------------------
print("\nMissing values:\n", df.isnull().sum())

# (Optional: Handle missing values if present)
# Example:
# df.fillna(method='ffill', inplace=True)

# -------------------------------
# 4. Encode categorical features
# -------------------------------
label_encoder = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

print("\nCategorical columns encoded successfully")

# -------------------------------
# 5. (Optional) Drop unwanted feature
# -------------------------------
# Example: remove one feature if needed
# df.drop('Feature_Name', axis=1, inplace=True)

# -------------------------------
# 6. Save FINAL processed dataset ✅
# -------------------------------
df.to_csv("../outputs/processed_loan_data.csv", index=False)
print("\nProcessed dataset saved successfully")

# -------------------------------
# 7. Features and target
# -------------------------------
X = df.drop('Default', axis=1)
y = df['Default']

# -------------------------------
# 8. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 9. Feature scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 10. Handle imbalance using SMOTE
# -------------------------------
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nBefore SMOTE:\n", y_train.value_counts())
print("\nAfter SMOTE:\n", pd.Series(y_train_resampled).value_counts())

# -------------------------------
# DONE
# -------------------------------
print("\n✅ PHASE 1 COMPLETED SUCCESSFULLY")