import numpy as np
from src.engines.yolo_logic import YOLOEngine
from src.core.types import TradeAction

def test_weighted_fusion():
    print("🚀 [TEST] Phase 22: Weighted Signal Fusion")
    
    engine = YOLOEngine()
    
    # 1. Mock Image
    dummy_img = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # 2. Run Analysis
    print("Running Analysis on Simulated Engulfing + Support...")
    signal = engine.analyze(dummy_img)
    
    print(f"\n[Result]")
    print(f"Action: {signal.action}")
    print(f"Confidence: {signal.confidence:.4f}")
    print(f"Reasons: {signal.reasoning}")
    print(f"Score Tag: {signal.ts_bias}")
    
    # 3. Assertions
    assert signal.action == TradeAction.BUY
    
    # Check Confidence components logic
    # Expected > 0.80 due to Confluence
    assert signal.confidence > 0.80, f"Confidence {signal.confidence} too low for Engulfing+Support"
    
    # Check Evidence
    reasons_str = str(signal.reasoning)
    assert "Engulfing" in reasons_str
    assert "Support Zone" in reasons_str
    
    print("✅ Fusion Logic Verified: High Confidence generated from Pattern + Context.")

if __name__ == "__main__":
    test_weighted_fusion()
