from pyspark.sql import SparkSession
import mlflow

from src.ingestion.spark_ingest import create_spark_session, ingest_scada, feature_engineering, label_anomalies
from src.models.isolation_forest import train_isolation_forest, FEATURES
from src.models.lstm_autoencoder import train_lstm_autoencoder, create_sequences, get_reconstruction_errors
from src.evaluation.metrics import time_aware_evaluation
import torch

spark = create_spark_session()

# 1. Ingest + feature engineer
df_spark = ingest_scada(spark, "data/raw/T1.csv")
df_spark = feature_engineering(df_spark)
df_spark = label_anomalies(df_spark)
df_pandas = df_spark.toPandas()

# 2. Train Isolation Forest
if_preds, if_scores, if_model = train_isolation_forest(df_pandas)

# 3. Evaluate IF
time_aware_evaluation(
    df_pandas["anomaly"].values, if_scores,
    df_pandas["timestamp"], run_name="IF_evaluation"
)

# 4. Train LSTM AE
lstm_model, scaler = train_lstm_autoencoder(df_pandas, seq_len=48, epochs=50)

# 5. Evaluate LSTM AE
from src.models.lstm_autoencoder import create_sequences
import numpy as np
from sklearn.preprocessing import StandardScaler

X = scaler.transform(df_pandas[FEATURES].dropna().values)
sequences = create_sequences(X, seq_len=48)
errors = get_reconstruction_errors(lstm_model, torch.FloatTensor(sequences))

# Align labels (sequences start at index seq_len)
labels_aligned = df_pandas["anomaly"].values[48:]
time_aware_evaluation(
    labels_aligned, errors,
    df_pandas["timestamp"].iloc[48:], run_name="LSTM_AE_evaluation"
)

print("Pipeline complete. View results at http://localhost:5000")