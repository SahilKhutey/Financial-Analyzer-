from src.engines.sentiment import SentimentEngine
from src.engines.fusion import SignalFusion
from src.core.types import VisionOutput, MarketBias, FinalSignal, TradeAction

def test_phase31_sentiment():
    print("🚀 [TEST] Phase 31: Market Sentiment")
    
    # 1. Test Sentiment Engine
    print("Testing News Scanner...")
    se = SentimentEngine()
    out = se.analyze("BTC")
    
    print(f"Sentiment Score: {out.score}")
    print(f"Bias: {out.bias}")
    print(f"Headlines: {len(out.headlines)}")
    
    assert -1.0 <= out.score <= 1.0
    assert len(out.headlines) == 3
    print("✅ Sentiment Analysis Verified")
    
    # 2. Test Fusion Integration
    # We want to see if Fusion accepts the sentiment object
    print("Testing Fusion Integration...")
    fusion = SignalFusion()
    
    # Mock Inputs
    # VisionOutput(market_bias, breakout_prob, reversal_prob, momentum_score, patterns_detected)
    mock_vision = VisionOutput(MarketBias.BULLISH, 0.8, 0.2, 0.7, ["MockPattern"])
    
    # Case A: Neutral Sentiment
    # just run it
    res = fusion.fuse(mock_vision, sentiment=out)
    print(f"Fusion Result (Normal): {res.action} | Conf: {res.confidence}")
    assert "News" in res.model_votes
    
    # Case B: Extreme Negative Sentiment (Veto Logic)
    # We manually force a "Crash" scenario
    from src.engines.sentiment import SentimentOutput
    crash_sent = SentimentOutput(score=-0.8, bias="BEARISH", headlines=[])
    
    res_crash = fusion.fuse(mock_vision, sentiment=crash_sent)
    print(f"Fusion Result (Crash Scenario): {res_crash.action} | Conf: {res_crash.confidence}")
    
    # Should NOT be BUY, or very low confidence if BUY
    # Logic: if score < -0.6 and raw > 0 -> raw=0.0
    assert res_crash.action != TradeAction.BUY or res_crash.confidence < 0.2
    
    print("✅ Fusion Logic Verified")
    print("\n✅ Phase 31 Tests Passed!")

if __name__ == "__main__":
    test_phase31_sentiment()
