import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURES = [
    "wind_speed_m_s", "lv_activepower_kw", "theoretical_power_curve_kwh",
    "power_deviation", "capacity_factor",
    "power_rolling_mean_1h", "power_rolling_std_1h",
    "power_delta", "hour", "month"
]

def train_isolation_forest(df_pandas: pd.DataFrame, contamination=0.05):
    mlflow.set_experiment("wind-turbine-anomaly")
    
    with mlflow.start_run(run_name="isolation_forest"):
        mlflow.log_params({
            "model": "IsolationForest",
            "contamination": contamination,
            "n_estimators": 200,
            "features": FEATURES
        })
        
        X = df_pandas[FEATURES].dropna()
        
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    IsolationForest(
                n_estimators=200,
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        pipe.fit(X)
        
        # Predictions: -1 = anomaly, 1 = normal → remap to 1/0
        preds = (pipe.predict(X) == -1).astype(int)
        scores = -pipe.named_steps["clf"].score_samples(
            pipe.named_steps["scaler"].transform(X)
        )
        
        mlflow.sklearn.log_model(pipe, "isolation_forest_model",
                                  registered_model_name="WindTurbineIF")
        
        return preds, scores, pipe