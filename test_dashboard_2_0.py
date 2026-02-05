import pytest
from src.core.types import FinalSignal, TradeAction, VisionOutput, MarketBias
from src.engines.fusion import SignalFusion
from src.engines.explanation import ExplanationEngine

def test_dashboard_features():
    print("🚀 [TEST] Phase 23: Dashboard 2.0 Backend")
    
    # 1. Setup Engines
    fusion = SignalFusion()
    expl = ExplanationEngine()
    
    # 2. Create Mock Inputs
    v_out = VisionOutput(MarketBias.BULLISH, 0.8, 0.1, 0.9, [])
    # Partial mock for sub-signals
    ml_mock = FinalSignal(TradeAction.BUY, 0.85, ["ML Trend"], "ML:0.85", "N/A", "Trending", model_votes=None)
    
    # 3. Test Fusion (Model Votes)
    print("Testing Fusion Votes...")
    final_sig = fusion.fuse(v_out, ml_sig=ml_mock)
    
    votes = final_sig.model_votes
    print(f"Votes: {votes}")
    
    assert votes is not None, "Model Votes should be populated"
    assert votes['Vision'] == "Bullish"
    assert votes['ML'] == "BUY"
    assert votes['Math'] == "N/A"
    print("✅ Logic Correct: Votes are tracking.")
    
    # 4. Test Explanation Checklist
    print("\nTesting Explanation Checklist...")
    checks = expl.generate_checklist(final_sig)
    print(f"Checklist: {checks}")
    
    assert len(checks) == 4, "Expected 4 standard checks"
    assert checks[0]['label'] == "Bullish Trend"
    print("✅ Logic Correct: Checklist generation.")
    
    # 5. Test AI Prompt
    print("\nTesting AI Prompt...")
    prompt = expl.generate_ai_prompt(final_sig, "BTC-USD")
    print(f"Prompt Head:\n{prompt[:100]}...")
    
    assert "SIGNAL: BUY" in prompt
    assert "VOTES:" in prompt
    print("✅ Logic Correct: AI Prompt constructed.")
    
    print("\n✅ Phase 23 Tests Passed!")

if __name__ == "__main__":
    test_dashboard_features()
