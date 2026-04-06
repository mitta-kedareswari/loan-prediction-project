import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.combine import SMOTETomek
import joblib

# Load data
df = pd.read_csv("data/Loan_default.csv")

# Drop ID columns
df.drop(columns=['LoanID', 'Loan_ID', 'Id'], inplace=True, errors='ignore')

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop('Default', axis=1)
y = df['Default']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SMOTE + Tomek (better class balancing)
smote = SMOTETomek(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Models
lr = LogisticRegression(max_iter=1000)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight='balanced',
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=7,
    scale_pos_weight=2,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    eval_metric='logloss',
    random_state=42
)

# Train models
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Predictions
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

# XGBoost threshold tuning
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_prob_xgb > 0.42).astype(int)

roc_auc = roc_auc_score(y_test, y_prob_xgb)
print("XGBoost ROC AUC:", round(roc_auc,4))

# Store metrics
results = []

def store_metrics(model_name, y_true, y_pred):
    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1_Score": round(f1_score(y_true, y_pred), 4)
    })

store_metrics("Logistic Regression", y_test, y_pred_lr)
store_metrics("Random Forest", y_test, y_pred_rf)
store_metrics("XGBoost", y_test, y_pred_xgb)

# Save metrics
metrics_df = pd.DataFrame(results)
metrics_df.to_csv("outputs/model_evaluation_metrics.csv", index=False)

# Save models
joblib.dump(lr, "models/logistic_regression.pkl")
joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(xgb, "models/xgboost.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Phase 3 completed")
print("Metrics saved in outputs/model_evaluation_metrics.csv")
print("Models saved successfully")