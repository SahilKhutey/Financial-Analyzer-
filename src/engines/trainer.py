"""
Step 30: Model Trainer Engine.
Handles loading user data and retraining the XGBoost model.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import joblib
from src.core.journal import TRAINING_DATA_PATH
from src.core.logger import SystemLogger

MODEL_PATH = "user_data/models/xgb_v2.json"
MODEL_DIR = "user_data/models"

class ModelTrainer:
    """
    Manages the lifecycle of user-personalized models.
    """
    def __init__(self):
        self.logger = SystemLogger()
        self._ensure_dir()
        
    def _ensure_dir(self):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR, exist_ok=True)
            
    def get_dataset_stats(self):
        """Return counts of collected samples."""
        if not os.path.exists(TRAINING_DATA_PATH):
            return {"count": 0, "wins": 0, "losses": 0}
            
        try:
            df = pd.read_csv(TRAINING_DATA_PATH)
            # Filter only completed (Win/Loss) for valid training count, 
            # though we might train on all if we had weak labels, but usually we need outcome.
            # For now, just return raw size
            return {
                "count": len(df),
                "wins": len(df[df['Result'] == 'WIN']),
                "losses": len(df[df['Result'] == 'LOSS'])
            }
        except:
             return {"count": 0, "wins": 0, "losses": 0}

    def train_model(self) -> str:
        """
        Retrains XGBoost on captured data.
        Returns status message.
        """
        if not os.path.exists(TRAINING_DATA_PATH):
            return "No training data found."
            
        df = pd.read_csv(TRAINING_DATA_PATH)
        
        # Filter: We can only learn from completed trades (WIN/LOSS)
        # We assume 'Result' column is populated (by user or auto-verifier)
        valid_df = df[df['Result'].isin(['WIN', 'LOSS'])].copy()
        
        if len(valid_df) < 10:
            return f"Not enough labeled data ({len(valid_df)} samples). Need > 10."
            
        # Prepare Data
        # Map Result to Target: WIN=1, LOSS=0
        valid_df['target'] = valid_df['Result'].map({'WIN': 1, 'LOSS': 0})
        
        # Feat Cols (must match ML Engine expectation)
        feat_cols = ['log_ret', 'vol', 'mom', 'rsi', 'vol_atr']
        
        # Check if cols exist
        missing = [c for c in feat_cols if c not in valid_df.columns]
        if missing:
             # Try to calc or fail? For now fail
             return f"Missing features in data: {missing}"
             
        X = valid_df[feat_cols]
        y = valid_df['target']
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=4, 
            learning_rate=0.05,
            eval_metric='logloss'
        )
        
        try:
            model.fit(X, y)
            
            # Save
            model.save_model(MODEL_PATH)
            
            self.logger.info(f"Retrained Model saved to {MODEL_PATH}")
            return f"✅ Success! Trained on {len(valid_df)} samples. Saved to {MODEL_PATH}"
            
        except Exception as e:
            self.logger.error(f"Training Failed: {e}")
            return f"Training Failed: {e}"
