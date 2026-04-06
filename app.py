from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load Models
MODEL_DIR = "models"

logistic_model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    try:
        features = [
            data["age"],
            data["income"],
            data["loan_amount"],
            data["credit_score"],
            data["months_employed"],
            data["interest_rate"],
            data["loan_term"],
        ]

        input_array = np.array(features).reshape(1, -1)

        # Use XGBoost as main model
        prediction = int(xgb_model.predict(input_array)[0])

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
