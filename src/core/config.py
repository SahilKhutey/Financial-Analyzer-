"""
Global Configuration Settings.
Centralized control for model parameters, paths, and thresholds.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Vision Model Settings
VISION_MODEL_NAME = "EfficientNet-B0 (Simulated)"
IMAGE_SIZE = (224, 224)

# Time Series Settings
TS_LOOKBACK_WINDOW = 50  # Candles
TS_FEATURES = ['log_returns', 'volatility', 'volume_delta']

# Risk Control
CONFIDENCE_THRESHOLD = 0.65
MAX_DAILY_DRAWDOWN_LIMIT = 0.02

# Fiduciary
ENABLE_REAL_TRADING = False  # HARD LOCK
