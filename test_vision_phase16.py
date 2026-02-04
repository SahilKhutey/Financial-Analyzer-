import pandas as pd
from src.engines.fusion import SignalFusion
from src.core.types import VisionOutput, MarketBias, TradeAction

def test_vision_fusion():
    print("🚀 [TEST] Phase 16: Vision-Fusion Integration")
    
    # 1. Setup Fusion Engine
    fusion = SignalFusion()
    
    # 2. Create Vision-Only Input
    # Strong Bullish Vision
    vision_out = VisionOutput(
        market_bias=MarketBias.BULLISH,
        breakout_prob=0.8,
        reversal_prob=0.2,
        momentum_score=0.9,
        patterns_detected=["Bullish Engulfing"]
    )
    
    # 3. Fuse with NO Data Inputs (Vision Only)
    print("Testing Vision-Only Fusion...")
    sig = fusion.fuse(
        vision=vision_out,
        ts=None,
        regime=None,
        ml_sig=None,
        deep_sig=None,
        math_sig=None
    )
    
    print(f"Action: {sig.action}")
    print(f"Confidence: {sig.confidence}")
    print(f"Reasons: {sig.reasoning}")
    print(f"TS Bias Tag: {sig.ts_bias}")
    
    # 4. Assertions
    # Score should be 1.0 * 0.8 (breakout) = 0.8
    # Since weight is 100% vision, raw_signal = 0.8
    # Confidence = 0.8
    
    assert sig.action == TradeAction.BUY, f"Expected BUY, got {sig.action}"
    assert sig.confidence == 0.8, f"Expected 0.8, got {sig.confidence}"
    assert "Vision Only" in sig.ts_bias, "Missing 'Vision Only' tag in bias"
    assert "⚠️ Vision Only Mode" in str(sig.reasoning), "Missing warning in reasoning"
    
    print("✅ Phase 16 Tests Passed!")

if __name__ == "__main__":
    test_vision_fusion()
