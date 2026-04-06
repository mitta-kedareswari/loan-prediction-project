import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load saved metrics
metrics_df = pd.read_csv("outputs/model_evaluation_metrics.csv")

# Models and metrics
models = metrics_df["Model"]
metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]

# X-axis positions
x = np.arange(len(models))
width = 0.2

# Create figure
plt.figure(figsize=(10,6))

# Plot bars
plt.bar(x - 1.5*width, metrics_df["Accuracy"], width, label="Accuracy", color="skyblue")
plt.bar(x - 0.5*width, metrics_df["Precision"], width, label="Precision", color="orange")
plt.bar(x + 0.5*width, metrics_df["Recall"], width, label="Recall", color="green")
plt.bar(x + 1.5*width, metrics_df["F1_Score"], width, label="F1 Score", color="red")

# Labels and formatting
plt.xticks(x, models, rotation=15)
plt.ylabel("Score")
plt.xlabel("Models")
plt.title("Model Performance Comparison")
plt.ylim(0,1)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

# Save chart
plt.savefig("outputs/model_metrics_comparison.png", dpi=300)

# Show chart
plt.show()

print("Metrics comparison chart saved in outputs/model_metrics_comparison.png")
