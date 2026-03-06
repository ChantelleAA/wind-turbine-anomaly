# Wind Turbine Anomaly Detection Pipeline

End-to-end anomaly detection pipeline for **multivariate wind turbine SCADA time-series** (batch + inference), built as part of my PhD work in AI for offshore wind energy.
                                                                                                                                                 
This repo focuses on:
- scalable ingestion/feature engineering (Spark-friendly)
- anomaly detection (Isolation Forest baseline + LSTM Autoencoder)
- time-series-aware evaluation metrics
- experiment tracking + model registry with MLflow
- Dockerised inference API for reproducible deployment

Repo: https://github.com/ChantelleAA/wind-turbine-anomaly

---

## Architecture

![Architecture](assets/architecture.png)

High-level flow:
1. **SCADA CSV** → Spark ingestion & cleaning
2. Feature engineering (rolling stats, lags, time features)
3. Train models (baseline IF + deep LSTM-AE)
4. Evaluate with time-series-aware metrics
5. Track + register models with **MLflow**
6. Serve predictions via **Flask API** in **Docker**

---

## What’s inside

```

.
├── data/
│   └── raw/                  # raw SCADA CSVs (example path used in main.py)
├── docker/
│   └── Dockerfile            # containerised inference service
├── mlflow/                   # MLflow-related assets (tracking/registry)
├── src/
│   ├── ingestion/
│   │   └── spark_ingest.py    # Spark session, ingestion, feature engineering, rule-based labels
│   ├── models/
│   │   └── lstm_autoencoder.py# PyTorch LSTM Autoencoder training + reconstruction errors
│   ├── evaluation/
│   │   └── metrics.py         # PR-AUC, ROC-AUC, point-adjusted F1, detection latency
│   └── serve/
│       └── inference_api.py   # Flask API that loads MLflow-registered model
├── main.py                    # pipeline runner (batch)
└── requirements.txt

````

Key scripts:
- `src/ingestion/spark_ingest.py` for ingestion + features + optional rule-based labels :contentReference[oaicite:0]{index=0}  
- `src/models/lstm_autoencoder.py` for LSTM AE training + reconstruction scoring :contentReference[oaicite:1]{index=1}  
- `src/evaluation/metrics.py` for time-series-aware evaluation (point-adjusted F1 + latency) :contentReference[oaicite:2]{index=2}  
- `src/serve/inference_api.py` for MLflow-powered inference API :contentReference[oaicite:3]{index=3}  

---

## Data expectations

This pipeline assumes SCADA data as a CSV with (at minimum) columns consistent with:
- `date_time` (parsed into a `timestamp`)
- `wind_speed_m_s`
- `lv_activepower_kw`
- `theoretical_power_curve_kwh`

Ingestion standardises column names and parses timestamps in:
`dd MM yyyy HH:mm` format. :contentReference[oaicite:4]{index=4}

> Note: Many real SCADA datasets differ by vendor. If your headers differ, update the column mappings in `spark_ingest.py`.

---

## Feature engineering (what we compute)

Implemented features include: :contentReference[oaicite:5]{index=5}
- rolling mean/std of power (1h window)
- rolling mean of wind speed (24h window)
- **power deviation** = actual power − theoretical power curve
- **capacity factor** (currently a simple normalisation; can be updated to rated power)
- time features (hour, day of week, month)
- lag + delta for power to detect sudden drops

---

## Labels (optional)

If you don’t have labeled faults, the pipeline can create **rule-based anomaly labels** using domain heuristics (power far below theoretical, negative power, sudden drops). :contentReference[oaicite:6]{index=6}

These are meant as a weak “silver label” for:
- quick evaluation
- sanity checks during development

---

## Models

### Model A — Isolation Forest (baseline)
- Fast unsupervised baseline, suitable for noisy/imbalanced sensor readings.
- Inference is designed to load a **registered MLflow model** named `WindTurbineIF` (Production stage). :contentReference[oaicite:7]{index=7}

> If you haven’t registered the IF model yet, the API will fail to load. See “MLflow setup” below.

### Model B — LSTM Autoencoder (deep)
- Learns to reconstruct normal behavior; large reconstruction error implies anomaly.
- Implemented in `src/models/lstm_autoencoder.py`. :contentReference[oaicite:8]{index=8}

---

## Evaluation (time-series aware)

Standard metrics can be misleading on temporal, imbalanced anomaly data.
This repo includes: :contentReference[oaicite:9]{index=9}
- PR-AUC (preferred for imbalanced)
- ROC-AUC
- “best threshold” by maximising F1
- **point-adjusted F1** (counts an anomaly window as detected if any point is detected)
- detection latency (how quickly anomalies are flagged within a window)

---

## Quickstart (local)

### 1) Create an environment and install deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

`requirements.txt` includes Flask, numpy, pandas, pyspark, scikit-learn, torch. ([GitHub][1])

### 2) Run the batch pipeline

```bash
python main.py
```

By default, `main.py` expects `data/raw/T1.csv`. ([GitHub][2])

---

## MLflow setup (local dev)

You can run a local MLflow server (simple dev mode):

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Then point your tracking URI (optional):

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

> Databricks note: In production, you’d typically use Databricks MLflow + Unity Catalog model registry.

---

## Inference API (Docker)

The Docker image serves a Flask API on port `8080` and loads a **Production** model from MLflow registry: `models:/WindTurbineIF/Production`. ([GitHub][3])

### Build & run

```bash
docker build -f docker/Dockerfile -t wind-turbine-anomaly .
docker run -p 8080:8080 wind-turbine-anomaly
```

### Endpoints

* `GET /health` → `{ "status": "ok" }` ([GitHub][3])
* `POST /predict` → returns anomaly flags + scores ([GitHub][3])

Example request:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "wind_speed_m_s": 7.2,
        "lv_activepower_kw": 820,
        "theoretical_power_curve_kwh": 900,
        "power_deviation": -80,
        "capacity_factor": 0.23,
        "power_rolling_mean_1h": 810,
        "power_rolling_std_1h": 40,
        "power_delta": -10,
        "hour": 14,
        "month": 3
      }
    ]
  }'
```

---

## Notes / known gaps (current repo state)

* `main.py` references an Isolation Forest training module (`src.models.isolation_forest`) that is not currently present in the repo, so you’ll want to add/commit that module or adjust `main.py` to match your actual IF training code. ([GitHub][2])
* Several files are currently committed as single-line scripts; you may want to run a formatter (e.g., `black`) for readability.

---

## Roadmap

* [ ] Add/commit Isolation Forest training script and register to MLflow (`WindTurbineIF`)
* [ ] Add dataset schema notes + column mapping configs for different SCADA vendors
* [ ] Add notebook or report showing model performance on a sample dataset
* [ ] Add CI (lint + tests) and pre-commit hooks
* [ ] Add drift monitoring / data quality checks

---

## Skills / keywords (for recruiters)

PySpark · Databricks · Delta Lake · MLflow · Time Series · Anomaly Detection · Isolation Forest · LSTM Autoencoder · PyTorch · Docker · Flask · Python

---

## License

MIT (or add your preferred license)
