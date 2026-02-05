from src.engines.macro import MacroEngine, MacroState
from src.engines.regime import RegimeFilter
import pandas as pd
import numpy as np

def test_phase32_macro():
    print("🚀 [TEST] Phase 32: Global Macro Engine")
    
    # 1. Test Macro Engine
    print("Testing Macro Monitor...")
    me = MacroEngine()
    state = me.analyze()
    
    print(f"DXY: {state.dxy}")
    print(f"VIX: {state.vix}")
    print(f"Regime: {state.regime}")
    print(f"Multiplier: {state.multiplier}x")
    
    assert 0.0 <= state.multiplier <= 1.25
    print("✅ Macro Engine Verified")
    
    # 2. Test Regime Filter Integration
    print("Testing Regime Integration...")
    rf = RegimeFilter()
    
    # Create Dummy Data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    df = pd.DataFrame({
        'Open': np.linspace(100, 110, 100),
        'High': np.linspace(100, 110, 100) + 0.5, # Small spread
        'Low': np.linspace(100, 110, 100) - 0.5,
        'Close': np.linspace(100, 110, 100) + 0.1,
        'Volume': np.random.randint(1000, 5000, 100) # Liquid
    }, index=dates)
    
    # A. Normal Case
    normal_macro = MacroState(100.0, 15.0, 4.0, "RISK_ON", 1.25)
    res_norm = rf.analyze(df, macro_state=normal_macro)
    print(f"Normal Result: {res_norm.is_safe} ({res_norm.state})")
    assert res_norm.is_safe == True
    
    # B. Recession Case (Veto)
    recession_macro = MacroState(110.0, 45.0, 3.5, "RECESSION", 0.0)
    res_rec = rf.analyze(df, macro_state=recession_macro)
    print(f"Recession Result: {res_rec.is_safe} ({res_rec.reason})")
    
    assert res_rec.is_safe == False
    assert "RECESSION" in res_rec.state
    assert "Risk Veto" in res_rec.reason
    
    # C. Risk-Off Tagging
    risk_off_macro = MacroState(106.0, 28.0, 4.2, "RISK_OFF", 0.5)
    res_off = rf.analyze(df, macro_state=risk_off_macro)
    print(f"Risk-Off Result: {res_off.state}")
    
    assert "RISK-OFF" in res_off.state
    
    print("✅ Regime Integration Verified")
    print("\n✅ Phase 32 Tests Passed!")

if __name__ == "__main__":
    test_phase32_macro()
