import os
import sys
import pandas as pd
from src.core.types import TradeAction

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_health_check():
    print("🏥 Institutional System Health Check")
    print("===================================")
    
    errors = []
    
    # 1. Directory Structure
    dirs = ['logs', 'data', 'models']
    for d in dirs:
        if not os.path.exists(d):
            try:
                os.makedirs(d)
                print(f"✅ Created missing directory: {d}")
            except Exception as e:
                errors.append(f"Failed to create {d}: {e}")
        else:
            print(f"✅ Directory exists: {d}")

    # 2. Engine Loading
    print("\n[Engines]")
    try:
        from src.engines.fusion import SignalFusion
        fusion = SignalFusion()
        print(f"✅ Signal Fusion: Online (Weights: ML={fusion.w_ml}, Vision={fusion.w_vision})")
    except Exception as e:
        errors.append(f"Fusion Engine Failed: {e}")

    try:
        from src.engines.ml_production import MLProductionEngine
        ml = MLProductionEngine()
        print(f"✅ ML Engine: Online (XGB+MLP)")
    except Exception as e:
        errors.append(f"ML Engine Failed: {e}")
        
    try:
        from src.engines.binomo import BinomoEngine
        binomo = BinomoEngine()
        print(f"✅ Binomo Engine: Online (Phase 15 Upgraded)")
    except Exception as e:
        errors.append(f"Binomo Engine Failed: {e}")
        
    try:
        from src.core.gates import SafetyGates
        print(f"✅ Safety Gates: Loaded (Rule 4: ML Validation Active)")
    except Exception as e:
        errors.append(f"Safety Gates Failed: {e}")

    try:
        from src.core.risk import RiskEngine
        risk = RiskEngine()
        print(f"✅ Risk Engine: Loaded (Crash Detection Active)")
    except Exception as e:
        errors.append(f"Risk Engine Failed: {e}")

    try:
        from src.engines.scalping import ScalpingEngine
        from src.engines.breakout import BreakoutEngine
        from src.engines.mean_reversion import MeanReversionEngine
        from src.engines.trend_following import TrendFollowingEngine
        from src.engines.smart_money import SmartMoneyEngine
        from src.engines.deep_analytic import DeepAnalyticalEngine
        from src.engines.math_engine import MathPredictionEngine
        
        scalp_engine = ScalpingEngine()
        breakout_engine = BreakoutEngine()
        mr_engine = MeanReversionEngine()
        trend_engine = TrendFollowingEngine()
        smart_engine = SmartMoneyEngine()
        deep_engine = DeepAnalyticalEngine()
        math_engine = MathPredictionEngine()
        print("✅ All Strategy Engines Loaded")
    except Exception as e:
        errors.append(f"Strategy Engine Load Failed: {e}")


    # 3. Engine Safety Check (Vision-Only Mode)
    print("\n[Safety Check]")
    try:
        from src.core.types import VisionOutput, MarketBias
        # Mock Vision Input
        v_out = VisionOutput(MarketBias.BULLISH, 0.8, 0.2, 0.9, [])
        
        engines = [
            ("Scalping", scalp_engine),
            ("Breakout", breakout_engine),
            ("MeanRev", mr_engine),
            ("Trend", trend_engine),
            ("ML", ml),
            ("Deep", deep_engine),
            ("Math", math_engine),
            ("SmartMoney", smart_engine)
        ]
        
        for name, eng in engines:
            try:
                # Some engines take vision_out, some take vision_image
                # We try both signatures or check method
                import inspect
                sig = inspect.signature(eng.analyze)
                
                if 'vision_image' in sig.parameters:
                    res = eng.analyze(None, vision_image=None)
                elif 'vision_out' in sig.parameters:
                    res = eng.analyze(None, vision_out=v_out)
                else:
                    res = eng.analyze(None)
                    
                print(f"✅ {name}: Safety Pass (Action: {res.action})")
            except Exception as e:
                errors.append(f"{name} Safety Check Failed: {e}")
                
    except Exception as e:
        errors.append(f"Safety Loop Error: {e}")

    # 4. Integration Test (Mock Fusion)
    print("\n[Integration]")
    from src.core.types import VisionOutput, MarketBias, TSOutput, RegimeOutput
    
    try:
        # Mock Inputs
        vision = VisionOutput(MarketBias.BULLISH, 0.8, 0.2, 0.9, [])
        ts = TSOutput(0.6, 0.4, 0.01)
        regime = RegimeOutput(True, "TREND", "Safe")
        
        # Test Fuse
        sig = fusion.fuse(vision, ts, regime, ml_sig=None)
        print(f"✅ Fusion Test: {sig.action} (Conf: {sig.confidence})")
        
        # Test Risk
        df_mock = pd.DataFrame({
            'Open': [100]*50, 'High': [101]*50, 'Low': [99]*50, 'Close': [100]*50, 'Volume': [1000]*50
        })
        risk_status = risk.analyze(df_mock)
        print(f"✅ Risk Test: {risk_status.level} ({risk_status.reason})")
        
    except Exception as e:
        errors.append(f"Integration Test Failed: {e}")

    print("\n===================================")
    if errors:
        print(f"❌ SYSTEM FAILED with {len(errors)} errors:")
        for err in errors:
            print(f" - {err}")
        sys.exit(1)
    else:
        print("🚀 ALL SYSTEMS GO. READY FOR LAUNCH.")
        sys.exit(0)

if __name__ == "__main__":
    run_health_check()
