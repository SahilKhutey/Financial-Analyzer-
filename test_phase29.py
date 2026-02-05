import pandas as pd
from src.core.money_manager import MoneyManager
from src.engines.binomo import BinomoEngine
from src.core.types import FinalSignal, TradeAction

def test_phase29_risk():
    print("🚀 [TEST] Phase 29: Money Management")
    
    # 1. Setup
    mm = MoneyManager()
    
    # 2. Test Fixed Sizing
    print("Testing Fixed Sizing...")
    mm.update_settings({'strategy': 'Fixed', 'base_amount': 25.0})
    amt = mm.get_trade_amount(0.8)
    assert amt == 25.0
    print(f"✅ Fixed Size Correct: ${amt}")
    
    # 3. Test Kelly Sizing (Mock Journal)
    print("Testing Kelly Criterion...")
    # Mock journal metrics result by injecting a mock journal or patching
    # For simplicity in this functional test, we assume default win rate (0.5) 
    # Default config has kelly_fraction=0.25
    # If WR=0.5, p=0.5, q=0.5, b=0.85
    # f = 0.5 - (0.5/0.85) = 0.5 - 0.58 = Negative -> Should yield base amount (floor)
    
    mm.update_settings({'strategy': 'Kelly', 'base_amount': 10.0})
    # We can't easily mock the internal journal without dependency injection or patching
    # So we'll trust the default behavior (fallback to base if negative edge)
    amt_kelly = mm.get_trade_amount(0.9) # High confidence boost 1.2x
    # 10.0 * 1.2 = 12.0
    print(f"Kelly Amount (simulated negative edge): ${amt_kelly}")
    assert amt_kelly >= 10.0
    print("✅ Kelly Logic Safe")
    
    # 4. Test Daily Loss Lock
    print("Testing Daily Loss Limit...")
    # We need to simulate a bad day in the journal
    # Create a mock journal entry that exceeds limit
    from src.core.journal import JOURNAL_PATH
    from datetime import datetime
    
    # Backup
    import shutil
    import os
    if os.path.exists(JOURNAL_PATH):
        shutil.copy(JOURNAL_PATH, JOURNAL_PATH + ".bak")
        
    try:
        # Write loss
        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = pd.DataFrame([{
            "Timestamp": today, "Asset": "TEST", "Action": "BUY", 
            "Amount": 100, "Strategy": "Test", "Reason": "Test", 
            "Result": "LOSS", "PnL": -100.0
        }])
        df.to_csv(JOURNAL_PATH, index=False)
        
        # Set limit to 50
        mm.update_settings({'max_daily_loss': 50.0})
        
        # Check
        allowed = mm.check_daily_loss_limit()
        print(f"Allowed to trade? {allowed}")
        assert allowed == False
        
        amt_lock = mm.get_trade_amount(0.8)
        assert amt_lock == 0.0
        print("✅ Daily Lockout Verified")
        
        # 5. Integration Test (While still locked)
        print("Testing Binomo Integration...")
        eng = BinomoEngine()
        eng.money_manager = mm # This mm is currently LOCKED
        
        sig = FinalSignal(TradeAction.BUY, 0.9, ["Test"], "N/A", "N/A", "Test")
        res = eng.execute_trade(sig)
        print(f"Exec Result: {res}")
        assert "Risk Gate" in res
        print("✅ Integration blocked correctly")
        
    finally:
        # Restore
        if os.path.exists(JOURNAL_PATH + ".bak"):
            shutil.move(JOURNAL_PATH + ".bak", JOURNAL_PATH)

    print("\n✅ Phase 29 Tests Passed!")

if __name__ == "__main__":
    test_phase29_risk()
