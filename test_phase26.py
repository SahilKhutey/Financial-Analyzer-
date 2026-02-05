import pandas as pd
import numpy as np
from src.engines.optimizer import OptimizerEngine
from src.core.config import FUSION_WEIGHTS

def test_phase26_optimizer():
    print("🚀 [TEST] Phase 26: Intelligent Optimization")
    
    # 1. Setup
    opt = OptimizerEngine()
    
    # 2. Create Mock Data (need >200 rows)
    print("Generating Mock Data (300 candles)...")
    dates = pd.date_range(start="2023-01-01", periods=300, freq="H")
    df = pd.DataFrame(index=dates)
    df['Open'] = np.linspace(100, 200, 300)
    df['High'] = df['Open'] + 5
    df['Low'] = df['Open'] - 5
    df['Close'] = df['Open'] + np.random.normal(0, 2, 300) # Random noise
    df['Volume'] = 1000
    
    # Feature Engineering (Required by Engines)
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
    df['volatility'] = df['log_returns'].rolling(window=10).std().fillna(0)
    df['volume_delta'] = df['Volume'].pct_change().fillna(0)
    
    # Needs warm up > 50 for volatility
    df = df.iloc[50:].copy()
    
    # 3. Test Grid Generation
    print("\ntesting Grid Generation...")
    grid = opt.generate_grid()
    print(f"Grid Size: {len(grid)}")
    assert len(grid) >= 5
    assert 'ml' in grid[0]
    print("✅ Grid Logic Verified")
    
    # 4. Run Optimization (Simulated)
    print("\n🧠 Running Grid Search...")
    # This calls backtest 5 times, might take a few seconds
    results = opt.run_optimization(df, top_n=2)
    
    print(f"\nTop Result: {results[0]['name']}")
    print(f"Score: {results[0]['score']}")
    print(f"Weights: {results[0]['weights']}")
    
    assert len(results) == 2
    assert results[0]['score'] >= results[1]['score']
    print("✅ Rank Logic Verified")
    
    # 5. Check Fusion Update
    # Mock applying it
    best_weights = results[0]['weights']
    opt.backtester.fusion.set_weights(best_weights)
    assert opt.backtester.fusion.weights == best_weights
    print("✅ Weight Update Logic Verified")
    
    print("\n✅ Phase 26 Tests Passed!")

if __name__ == "__main__":
    test_phase26_optimizer()
