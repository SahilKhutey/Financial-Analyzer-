"""
Demo Script to Verify Institutional Engines.
Generates synthetic market data and runs:
1. MathPredictionEngine
2. DeepAnalyticalEngine
3. MLProductionEngine
"""

import pandas as pd
import numpy as np
from src.engines.math_engine import MathPredictionEngine
from src.engines.deep_analytic import DeepAnalyticalEngine
from src.engines.ml_production import MLProductionEngine

def generate_synthetic_data(n=500):
    """Generate trending/ranging synthetic OHLC data."""
    np.random.seed(42)
    prices = [100.0]
    dates = pd.date_range("2024-01-01", periods=n, freq="min")
    
    # Simulate a Trend then a Range
    for i in range(1, n):
        prev = prices[-1]
        if i < 300:
            # Trend Up
            change = np.random.normal(0.05, 0.2) 
        else:
            # Range / Volatile
            change = np.random.normal(0, 0.4)
            
        prices.append(prev + change)
        
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p + abs(np.random.normal(0, 0.1)) for p in prices],
        'Low': [p - abs(np.random.normal(0, 0.1)) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(100, 1000, n)
    })
    df.set_index('Date', inplace=True)
    return df

def run_demo():
    print("📉 Generating Synthetic Market Data (500 candles)...")
    df = generate_synthetic_data()
    print(f"Data Head:\n{df.head(3)}\n...")
    print(f"Data Tail:\n{df.tail(3)}\n")
    
    print("-" * 50)
    print("1️⃣ Testing Mathematical Prediction Engine...")
    math_engine = MathPredictionEngine()
    sig_math = math_engine.analyze(df)
    print(f"Signal: {sig_math.action.value} | Conf: {sig_math.confidence}")
    print(f"Reasoning: {sig_math.reasoning}")
    print(f"Bias: {sig_math.ts_bias}")
    
    print("-" * 50)
    print("2️⃣ Testing Deep Analytical Engine (Institutional)...")
    deep_engine = DeepAnalyticalEngine()
    sig_deep = deep_engine.analyze(df)
    print(f"Signal: {sig_deep.action.value} | Conf: {sig_deep.confidence}")
    print(f"Reasoning: {sig_deep.reasoning}")
    print(f"Regime: {sig_deep.regime}")
    
    print("-" * 50)
    print("3️⃣ Testing Enterprise ML Engine (XGB+MLP)...")
    ml_engine = MLProductionEngine()
    # First run triggers training
    print(">> Initializing & Training on history...")
    sig_ml = ml_engine.analyze(df) 
    print(f"Signal: {sig_ml.action.value} | Conf: {sig_ml.confidence}")
    print(f"Reasoning: {sig_ml.reasoning}")
    
    print("-" * 50)
    print("4️⃣ Testing Grand Fusion Engine...")
    from src.engines.fusion import SignalFusion
    from src.core.types import VisionOutput, TSOutput, RegimeOutput, MarketBias
    
    # Mock legacy inputs
    vision = VisionOutput(MarketBias.BULLISH, 0.6, 0.2, 0.7, [])
    ts = TSOutput(0.6, 0.4, 0.01)
    regime = RegimeOutput(True, "Trending", "Healthy")
    
    fusion = SignalFusion()
    final = fusion.fuse(vision, ts, regime, ml_sig=sig_ml, deep_sig=sig_deep, math_sig=sig_math)
    print(f"Grand Signal: {final.action.value} | Conf: {final.confidence}")
    print(f"Reasoning: {final.reasoning}")
    
    print("-" * 50)
    print("✅ Demo Complete.")

if __name__ == "__main__":
    run_demo()
