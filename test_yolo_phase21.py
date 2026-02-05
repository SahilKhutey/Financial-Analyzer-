import numpy as np
import pandas as pd
from src.engines.yolo_logic import YOLOEngine
from src.core.types import TradeAction

def test_yolo_advanced_logic():
    print("🚀 [TEST] Phase 21: YOLO Advanced Logic (Norm + Patterns)")
    
    engine = YOLOEngine()
    
    if not engine.sim_mode:
        print("⚠️ Warning: Using REAL model. Simulation test logic might fail.")
    else:
        print("✅ Running in Logic Simulation Mode")
        
    # 1. Mock Image
    dummy_img = np.zeros((600, 800, 3), dtype=np.uint8)
    h, w = dummy_img.shape[:2]
    
    # 2. Test Step A: Simulation Generation
    detections = engine._simulate_detection(dummy_img)
    print(f"Generated {len(detections)} mock objects (Candles + Context)")
    
    # 3. Test Step B: Normalization
    df = engine._normalize_sequence(detections, h)
    print("\n[Reconstructed OHLC]")
    print(df[['Label', 'Open', 'Close', 'High', 'Low']])
    
    assert len(df) == 3, "Expected 3 candles in simulation"
    
    # Verify Engulfing Math
    prev = df.iloc[-2]
    last = df.iloc[-1]
    
    print(f"\nChecking Engulfing Logic:")
    print(f"Prev (Bearish) Open: {prev['Open']:.3f}, Close: {prev['Close']:.3f} (Body Size: {abs(prev['Open']-prev['Close']):.3f})")
    print(f"Curr (Bullish) Open: {last['Open']:.3f}, Close: {last['Close']:.3f} (Body Size: {abs(last['Open']-last['Close']):.3f})")
    
    # Check simple math
    assert last['Close'] > prev['Open'], "Engulfing: Green Close should be > Red Open"
    assert last['Open'] < prev['Close'], "Engulfing: Green Open should be < Red Close"
    print("✅ Engulfing Math Verified")
    
    # 4. Test Step C: Integration
    signal = engine.analyze(dummy_img)
    print(f"\n[Final Signal]")
    print(f"Action: {signal.action}")
    print(f"Reasons: {signal.reasoning}")
    
    assert signal.action == TradeAction.BUY, "Expected BUY from Bullish Engulfing"
    assert "Bullish Engulfing" in str(signal.reasoning), "Reasoning must mention pattern"
    
    print("✅ Phase 21 Tests Passed!")

if __name__ == "__main__":
    test_yolo_advanced_logic()
