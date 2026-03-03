from flask import Flask, request, jsonify
import mlflow.sklearn
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load registered model from MLflow
model = mlflow.sklearn.load_model("models:/WindTurbineIF/Production")

FEATURES = [
    "wind_speed_m_s", "lv_activepower_kw", "theoretical_power_curve_kwh",
    "power_deviation", "capacity_factor",
    "power_rolling_mean_1h", "power_rolling_std_1h",
    "power_delta", "hour", "month"
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"]   # list of dicts
    df   = pd.DataFrame(data)[FEATURES]
    preds  = (model.predict(df) == -1).astype(int).tolist()
    scores = (-model.score_samples(df)).tolist()
    return jsonify({"anomalies": preds, "scores": scores})

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)