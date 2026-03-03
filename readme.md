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
```
