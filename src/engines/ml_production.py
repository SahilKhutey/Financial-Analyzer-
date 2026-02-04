"""
Enterprise ML Production Engine.
(XGBoost + Simulated LSTM/MLP + Calibration)

Architecture:
1. Feature Extraction
2. Strong Labeling (Future Max/Min > Threshold)
3. Sequence Learning (Windowed Data)
4. Ensemble Model (XGB + Neural Net)
5. Probability Calibration (Isotonic)
6. Edge-Only Trading Logic
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
from src.core.types import FinalSignal, TradeAction
import os

class MLProductionEngine:
    """
    Production-Grade ML Engine with Online Learning.
    """
    
    def __init__(self):
        self.lookback = 20 # Sequence length
        self.threshold = 0.0006 # Labeling threshold
        self.retrain_interval = 300 # Candles
        self.data_buffer = [] 
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_train_size = 0
        
        # 1. XGBoost Model (Structure)
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=300, # Reduced slightly for speed
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            n_jobs=1
        )
        
        # 2. Neural Net (Time Flow / "LSTM")
        # Python 3.14 does not support TensorFlow yet.
        # We use MLPClassifier on flattened sequences to approximate high-dim pattern matching.
        self.mlp_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='tanh', # Tanh often better for financial time series normalized data
            solver='adam',
            max_iter=200,
            early_stopping=True,
            random_state=42
        )
        
        # Calibrator placeholder (fit during training)
        self.calibrated_xgb = None
        
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()

    def _strong_labels(self, prices: pd.Series) -> np.ndarray:
        """
        Generate labels:
        1: Max future price > threshold (BUY)
        0: Min future price < -threshold (SELL) (Mapped to 0 for binary class)
        -1: NO TRADE
        """
        labels = []
        lookahead = 5
        vals = prices.values
        
        for i in range(len(vals) - lookahead):
            current = vals[i]
            future = vals[i+1 : i+lookahead+1]
            
            # Percentage change
            max_change = (np.max(future) - current) / current
            min_change = (current - np.min(future)) / current
            
            if max_change > self.threshold:
                labels.append(1) # Buy
            elif min_change > self.threshold:
                labels.append(0) # Sell
            else:
                labels.append(-1) # No Trade
                
        # Pad end
        labels.extend([-1] * lookahead)
        return np.array(labels)

    def _build_dataset(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare X (features), X_seq (flattened sequence), y (labels)
        """
        # Feature Engineering
        df = df.copy()
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['vol'] = df['log_ret'].rolling(10).std()
        df['mom'] = df['Close'].diff(5)
        
        # New Features (Refinement)
        df['rsi'] = self._calculate_rsi(df['Close'])
        atr = self._calculate_atr(df)
        df['vol_atr'] = df['vol'] / (atr / df['Close']) # Normalized Volatility
        df.dropna(inplace=True)
        
        # Features to use (Extended)
        feature_cols = ['log_ret', 'vol', 'mom', 'rsi', 'vol_atr']
        X_raw = df[feature_cols].values
        
        # Labels
        y_raw = self._strong_labels(df['Close'])
        
        # Create Sequences
        X_flat = []
        X_seq_flat = [] # MLP input (n_samples, window*n_features)
        y_valid = []
        
        # Need window history
        for i in range(self.lookback, len(X_raw)):
            label = y_raw[i]
            if label != -1: # Filter NO_TRADE for training
                # Flat features (current candle)
                X_flat.append(X_raw[i])
                
                # Sequence features (flattened)
                seq = X_raw[i-self.lookback : i]
                X_seq_flat.append(seq.flatten())
                
                y_valid.append(label)
                
        return np.array(X_flat), np.array(X_seq_flat), np.array(y_valid)

    def train(self, df: pd.DataFrame):
        """
        Train the models.
        """
        if len(df) < 200: return # Need min data
        
        X_flat, X_seq, y = self._build_dataset(df)
        if len(y) < 50: return # Need min labeled samples
        
        # Scale
        self.scaler.fit(X_seq) # Scale based on sequence distribution
        X_seq_scaled = self.scaler.transform(X_seq)
        
        # Train & Calibrate XGB (CV mode is more robust than prefit)
        # This fits the underlying XGB model on folds and calibrates
        self.calibrated_xgb = CalibratedClassifierCV(self.xgb_model, method='isotonic', cv=3)
        self.calibrated_xgb.fit(X_flat, y)
        
        # Train MLP
        self.mlp_model.fit(X_seq_scaled, y)
        
        self.is_trained = True
        print(f"ML Engine Trained on {len(y)} samples.")

    def analyze(self, df: pd.DataFrame) -> FinalSignal:
        """
        Run Live Prediction Pipeline.
        """
        if df is None:
             return FinalSignal(TradeAction.STAY_OUT, 0.0, ["ML: No Data"], "N/A", "N/A", "N/A")

        # Online Learning & Retraining
        # Check if we need initial training
        if not self.is_trained and len(df) > 200:
            self.train(df)
            self.last_train_size = len(df)
            
        # Check for periodic retraining (every 300 new candles)
        if self.is_trained and (len(df) - self.last_train_size) >= self.retrain_interval:
            print(f"♻️ Retraining ML Engine (New data: {len(df) - self.last_train_size} candles)...")
            self.train(df)
            self.last_train_size = len(df)

        if not self.is_trained:
            return FinalSignal(TradeAction.STAY_OUT, 0.0, ["Insufficient Data for ML"], "N/A", "N/A", "N/A")

        # Prepare Live Input
        # (Recreate features exactly as training)
        dataset = df.copy()
        dataset['log_ret'] = np.log(dataset['Close'] / dataset['Close'].shift(1))
        dataset['vol'] = dataset['log_ret'].rolling(10).std()
        dataset['mom'] = dataset['Close'].diff(5)
        
        # New Features (Refinement)
        dataset['rsi'] = self._calculate_rsi(dataset['Close'])
        atr = self._calculate_atr(dataset)
        dataset['vol_atr'] = dataset['vol'] / (atr / dataset['Close'])
        
        # Get last sequence
        if len(dataset) < self.lookback + 5: return FinalSignal(TradeAction.STAY_OUT, 0.0, ["Data < Lookback"], "N/A", "N/A", "N/A")
        
        # Current Features
        feat_cols = ['log_ret', 'vol', 'mom', 'rsi', 'vol_atr']
        current_feats = dataset[feat_cols].iloc[-1].values.reshape(1, -1)
        
        # Sequence Features
        # data[-lookback:] predictions for t+1
        seq_feats = dataset[feat_cols].iloc[-self.lookback:].values.flatten().reshape(1, -1)
        
        # Scale
        try:
            seq_scaled = self.scaler.transform(seq_feats)
        except:
            return FinalSignal(TradeAction.STAY_OUT, 0.0, ["Scaling Error"], "N/A", "N/A", "N/A")
            
        # Predict
        try:
            # P1: Calibrated XGB
            p1 = self.calibrated_xgb.predict_proba(current_feats)[0][1] # Prob of class 1 (Buy)
            
            # P2: MLP
            p2 = self.mlp_model.predict_proba(seq_scaled)[0][1]
            
            # Ensemble
            final_prob = 0.6 * p1 + 0.4 * p2
            
            # Decision
            action = TradeAction.STAY_OUT
            conf = 0.0
            reasons = []
            
            reasons.append(f"Ens P: {final_prob:.2f}")
            reasons.append(f"XGB: {p1:.2f} | MLP: {p2:.2f}")
            
            if final_prob > 0.68:
                action = TradeAction.BUY
                conf = final_prob
            elif final_prob < 0.32:
                action = TradeAction.SELL
                conf = 1.0 - final_prob
            else:
                conf = max(final_prob, 1-final_prob)
                
            return FinalSignal(
                action=action,
                confidence=round(conf, 2),
                reasoning=reasons,
                vision_bias="N/A",
                ts_bias=f"ML:{final_prob:.2f}",
                regime="N/A"
            )
            
        except Exception as e:
            return FinalSignal(TradeAction.STAY_OUT, 0.0, [f"ML Error: {str(e)}"], "N/A", "N/A", "N/A")
