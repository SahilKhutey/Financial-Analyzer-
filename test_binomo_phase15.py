import pandas as pd
import numpy as np
from src.engines.binomo import BinomoEngine
from src.core.types import FinalSignal, TradeAction

def test_binomo_phase15():
    print("🚀 [TEST] Phase 15: Binomo Engine Integration")
    
    # 1. Setup Data
    df = pd.DataFrame({
        'Open': [100.0, 101.0, 102.0],
        'High': [101.5, 102.5, 103.5],
        'Low': [99.5, 100.5, 101.5],
        'Close': [101.0, 102.0, 103.0],
        'Volume': [1000, 1500, 2000]
    })
    
    engine = BinomoEngine()
    
    # 2. Test Low Volatility -> 5 min Expiry
    # Volatility is approx (103.5-101.5)/102.0 = 0.019 (High)
    # Let's force low vol
    df_low = df.copy()
    # Vol needs to be > 0.0004 (Risk Limit) but < 0.0008 (Trend Limit)
    # Target 0.0006 implies range of 0.06 on price 100
    df_low['High'] = df_low['Open'] + 0.03
    df_low['Low'] = df_low['Open'] - 0.03
    df_low['Close'] = df_low['Open'] # Flat
    
    sig_low = engine.analyze(df_low)
    print(f"Low Vol Expiry: {sig_low.expiry}")
    # Calculation: (102.01 - 101.99) / 102 = 0.02 / 102 = 0.00019 < 0.0008 -> 5 min
    assert "5 min" in sig_low.expiry, f"Expected 5 min, got {sig_low.expiry}"
    
    # 3. Test Institutional Override
    # Create a Fusion Signal (Strong BUY)
    fusion_sig = FinalSignal(
        action=TradeAction.BUY,
        confidence=0.85,
        reasoning=["ML Strong Buy"],
        vision_bias="BULLISH",
        ts_bias="UP",
        regime="TRENDING"
    )
    
    # Even if local heuristic is weak (flat candle), Fusion should force BUY
    sig_override = engine.analyze(df_low, fusion_sig=fusion_sig)
    print(f"Override Action: {sig_override.action}")
    print(f"Override Reasons: {sig_override.reasoning}")
    
    assert sig_override.action == "BUY", "Fusion Override failed"
    assert "Institutional Override" in str(sig_override.reasoning), "Reasoning missing override tag"
    
    print("✅ Phase 15 Tests Passed!")

if __name__ == "__main__":
    test_binomo_phase15()
