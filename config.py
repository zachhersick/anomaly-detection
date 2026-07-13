from pathlib import Path
import os

DB_PATH = Path(os.getenv("DB_PATH", "anomaly_detection.db"))

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", 0.35))

RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
TRAIN_FRACTION = float(os.getenv("TRAIN_FRACTION", 0.60))
VALIDATION_FRACTION = float(os.getenv("VALIDATION_FRACTION", 0.20))
TEST_FRACTION = float(os.getenv("TEST_FRACTION", 0.20))
PURGE_GAP_STEPS = int(os.getenv("PURGE_GAP_STEPS", 50))

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
