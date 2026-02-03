"""
Demo Script for Enterprise Backtest.
"""
import pandas as pd
import numpy as np
from src.engines.backtest import BacktestEngine

def generate_synthetic_data(n=600):
    np.random.seed(42)
    prices = [100.0]
    dates = pd.date_range("2024-01-01", periods=n, freq="min")
    
    # Trend then Range
    for i in range(1, n):
        prev = prices[-1]
        change = np.random.normal(0.02, 0.2)
        if i < 400: change += 0.05 # Strong Up Trend
        
        prices.append(prev + change)
        
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p + 0.1 for p in prices],
        'Low': [p - 0.1 for p in prices],
        'Close': prices,
        'Volume': np.random.randint(100, 1000, n)
    })
    
    # Add legacy features
    df['volume_delta'] = df['Volume'].pct_change().fillna(0)
    
    df.set_index('Date', inplace=True)
    return df

def run_test():
    print("📉 Generating 600 candles of synthetic history...")
    df = generate_synthetic_data()
    
    print("🚀 Initializing Enterprise Backtest Engine...")
    bt = BacktestEngine()
    
    print("▶ Running Backtest (Mode: Auto Fusion)...")
    # This will take a moment as it loops and runs ML prediction for each bar
    trades = bt.run(df, strategy_mode="Auto (Fusion)")
    
    print("\n📜 Trade Log (Head):")
    print(trades.head() if not trades.empty else "No Trades")
    
    print("\n📊 Final Metrics:")
    metrics = bt.calculate_metrics(trades, 10000.0)
    for k,v in metrics.items():
        print(f"  {k}: {v}")
        
if __name__ == "__main__":
    run_test()
