import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("data/Loan_default.csv")

# Set style for better visuals
sns.set_style("whitegrid")

# -------------------------------
# Default Distribution
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Default', data=df, palette='Set2')
plt.title("Distribution of Loan Default")
plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/default_distribution.png", dpi=300)
plt.show()

# -------------------------------
# Age Distribution
# -------------------------------
plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=30, kde=True, color='skyblue')
plt.title("Age Distribution of Applicants")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/age_distribution.png", dpi=300)
plt.show()

# -------------------------------
# Income Distribution
# -------------------------------
plt.figure(figsize=(6,4))
sns.histplot(df['Income'], bins=30, kde=True, color='orange')
plt.title("Income Distribution of Applicants")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/income_distribution.png", dpi=300)
plt.show()

# -------------------------------
# Credit Score vs Default
# -------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x='Default', y='CreditScore', data=df, palette='Set3')
plt.title("Credit Score vs Loan Default")
plt.xlabel("Default")
plt.ylabel("Credit Score")
plt.tight_layout()
plt.savefig("outputs/credit_score_vs_default.png", dpi=300)
plt.show()

# -------------------------------
# Loan Amount vs Default
# -------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x='Default', y='LoanAmount', data=df, palette='Set1')
plt.title("Loan Amount vs Loan Default")
plt.xlabel("Default")
plt.ylabel("Loan Amount")
plt.tight_layout()
plt.savefig("outputs/loan_amount_vs_default.png", dpi=300)
plt.show()

# -------------------------------
# Correlation Heatmap
# -------------------------------
plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include='number').corr(),
            cmap='coolwarm',
            annot=False,
            linewidths=0.5)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=300)
plt.show()

print("EDA graphs saved successfully in outputs folder")