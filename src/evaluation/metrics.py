import numpy as np
import pandas as pd
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                              roc_auc_score, f1_score)
import mlflow

def time_aware_evaluation(y_true: np.ndarray, scores: np.ndarray,
                           timestamps: pd.Series, run_name: str):
    """
    Key insight: standard PR on imbalanced temporal data is misleading.
    We also compute:
      - PR-AUC (better than ROC for imbalanced data)
      - Point-adjust F1 (anomaly spans — if any point in a contiguous
        anomaly window is detected, the whole window counts as detected)
    """
    with mlflow.start_run(run_name=run_name, nested=True):
        # --- Standard metrics ---
        pr_auc = average_precision_score(y_true, scores)
        roc_auc = roc_auc_score(y_true, scores)
        
        # Best F1 across all thresholds
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
        best_f1 = np.max(f1_scores)
        
        # --- Point-Adjust F1 (time-series specific) ---
        pa_f1 = point_adjust_f1(y_true, scores > best_threshold)
        
        # --- Latency metric: how quickly is an anomaly detected? ---
        avg_detect_latency = detection_latency(y_true, scores > best_threshold)
        
        mlflow.log_metrics({
            "pr_auc":              pr_auc,
            "roc_auc":             roc_auc,
            "best_f1":             best_f1,
            "best_threshold":      best_threshold,
            "point_adjusted_f1":   pa_f1,
            "avg_detection_latency_steps": avg_detect_latency
        })
        
        print(f"PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | "
              f"Best F1: {best_f1:.4f} | PA-F1: {pa_f1:.4f}")
        
        return {"pr_auc": pr_auc, "pa_f1": pa_f1, "best_threshold": best_threshold}

def point_adjust_f1(y_true, y_pred):
    """
    If any predicted anomaly overlaps with a true anomaly segment,
    mark the entire true segment as detected.
    Standard practice in TSAD literature (Xu et al., 2022).
    """
    adjusted_pred = y_pred.copy()
    anomaly_segments = get_anomaly_segments(y_true)
    
    for start, end in anomaly_segments:
        if y_pred[start:end].any():
            adjusted_pred[start:end] = 1
    
    return f1_score(y_true, adjusted_pred)

def get_anomaly_segments(y_true):
    segments = []
    in_anomaly = False
    for i, v in enumerate(y_true):
        if v == 1 and not in_anomaly:
            start = i
            in_anomaly = True
        elif v == 0 and in_anomaly:
            segments.append((start, i))
            in_anomaly = False
    return segments

def detection_latency(y_true, y_pred):
    latencies = []
    for start, end in get_anomaly_segments(y_true):
        detected = np.where(y_pred[start:end])[0]
        if len(detected) > 0:
            latencies.append(detected[0])
    return np.mean(latencies) if latencies else float("inf")