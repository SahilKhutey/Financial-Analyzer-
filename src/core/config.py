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

# Auto-Optimization
FUSION_WEIGHTS = {
    'ml': 0.35,
    'deep': 0.25,
    'math': 0.15,
    'vision': 0.15,
    'ts': 0.10,
    'sentiment': 0.10
}

# Alerts & Monitoring
ENABLE_SOUND = True
DISCORD_WEBHOOK_URL = ""

# Risk & Money Management
risk_settings = {
    'strategy': 'Fixed', # Fixed, Percent, Kelly
    'base_amount': 10.0,
    'percent_equity': 0.05, # 5%
    'kelly_fraction': 0.25, # Quarter Kelly (Conservative)
    'max_daily_loss': 50.0
}
