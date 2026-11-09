import os
from pathlib import Path

SENSORS = ['temperature', 'pressure', 'vibration', 'flow_rate', 'voltage', 'current']

DB_PATH = Path(os.getenv("DB_PATH", "anomaly_detection.db"))

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.35"))

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TRAIN_FRACTION = float(os.getenv("TRAIN_FRACTION", "0.60"))
VALIDATION_FRACTION = float(os.getenv("VALIDATION_FRACTION", "0.20"))
TEST_FRACTION = float(os.getenv("TEST_FRACTION", "0.20"))
PURGE_GAP_STEPS = int(os.getenv("PURGE_GAP_STEPS", "50"))

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))

RAW_DATA_PATH = OUTPUT_DIR / "sensor_data_raw.csv"
FEATURES_PATH = OUTPUT_DIR / "sensor_data_features.csv"
RETENTION_PATH = OUTPUT_DIR / "feature_row_retention.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"
ALERTS_PATH = OUTPUT_DIR / "alerts.csv"
ALERT_EVENTS_PATH = OUTPUT_DIR / "alert_events.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
VALIDATION_RESULTS_PATH = OUTPUT_DIR / "validation_results.csv"
TEST_METRICS_PATH = OUTPUT_DIR / "test_metrics.csv"
ABLATION_RESULTS_PATH = OUTPUT_DIR / "ablation_results.csv"