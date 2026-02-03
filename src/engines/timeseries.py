"""
Step 3: Time-Series Engine.
Wraps an LSTM Model (Long Short-Term Memory) to analyze sequential data.

Simulation:
Since we lack a trained .h5 model, we will simulate the probabilistic output
using "feature-based heuristics" that mimic what a trained LSTM would likely output
given the technical structure (Momentum, Volatility, Trend).
"""

import pandas as pd
import numpy as np
from src.core.types import TSOutput
from src.core.config import TS_LOOKBACK_WINDOW
from src.data.processing import FeatureEngineer

class LSTMModel:
    """
    Simulated Time-Series Model.
    Analyzes a 50-candle window.
    """
    
    def __init__(self):
        self.lookback = TS_LOOKBACK_WINDOW
        
    def predict(self, df: pd.DataFrame) -> TSOutput:
        """
        Run inference on the sequence data.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            TSOutput: Probabilities and forecasts.
        """
        if len(df) < self.lookback:
            return TSOutput(0.5, 0.5, 0.0) # Insufficient data
            
        # 1. Feature Engineering (Simulating Deep Learning Feature extraction)
        feats = FeatureEngineer.compute_technical_features(df)
        last = feats.iloc[-1]
        
        # 2. Heuristic "Model Weights" simulation (Enhanced)
        score = 0
        
        # Trend (SMA)
        sma_20 = df['Close'].rolling(20).mean().iloc[-1]
        sma_50 = df['Close'].rolling(50).mean().iloc[-1]
        if sma_20 > sma_50: score += 0.3
        else: score -= 0.3
        
        # Momentum (RSI)
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            if rsi > 55: score += 0.2
            elif rsi < 45: score -= 0.2
            
        # MACD
        if 'macd_hist' in df.columns:
            hist = df['macd_hist'].iloc[-1]
            if hist > 0: score += 0.2
            else: score -= 0.2
            
        # Volume Confirmation
        vol_delta = df['volume_delta'].iloc[-1]
        if vol_delta > 0.2: score *= 1.1 # Amplify signal on volume spike
            
        # 3. Generate Probabilities (Sigmoid-like squash)
        # Base probability is 0.5, shifted by score (-1 to 1 range roughly)
        
        raw_bull = 0.5 + (score * 0.4) # Scale factor
        # Add some noise (uncertainty)
        noise = np.random.normal(0, 0.05)
        raw_bull += noise
        
        # Clip to 0-1
        bull_prob = max(0.01, min(0.99, raw_bull))
        bear_prob = 1.0 - bull_prob
        
        # 4. Volatility Forecast
        # Simple persistence model: forecast = current vol
        vol_forecast = last.get('volatility', 0.01) * 1.1 # Assume 10% expansion bias
        
        return TSOutput(
            bullish_prob=round(bull_prob, 2),
            bearish_prob=round(bear_prob, 2),
            forecast_volatility=round(vol_forecast, 4)
        )
