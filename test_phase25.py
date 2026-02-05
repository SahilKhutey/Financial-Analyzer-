import os
import shutil
import pandas as pd
from src.core.journal import Journal, JOURNAL_PATH
from src.engines.binomo import BinomoEngine
from src.core.types import FinalSignal, TradeAction

def test_phase25_journal():
    print("🚀 [TEST] Phase 25: Trade Journal & Persistence")
    
    # 1. Setup / Cleanup
    if os.path.exists(JOURNAL_PATH):
        # Backup existing to avoid messing up real user data if any
        shutil.copy(JOURNAL_PATH, JOURNAL_PATH + ".bak")
        os.remove(JOURNAL_PATH)
        print("ℹ️ Backed up existing journal.")

    try:
        # 2. Test Core Journal
        print("\n📝 Testing Core Logging...")
        journal = Journal()
        journal.log_trade("BTC", "BUY", 100.0, "Test", "Reason A")
        journal.log_trade("ETH", "SELL", 50.0, "Test", "Reason B")
        
        assert os.path.exists(JOURNAL_PATH), "Journal CSV not created"
        
        df = pd.read_csv(JOURNAL_PATH)
        print(f"Rows logged: {len(df)}")
        assert len(df) == 2
        assert df.iloc[0]['Asset'] == "BTC"
        assert df.iloc[0]['PnL'] == 0.0
        print("✅ Logging Verified")
        
        # 3. Test Integration
        print("\n⚡ Testing Binomo Integration...")
        eng = BinomoEngine()
        sig = FinalSignal(TradeAction.BUY, 0.8, ["Auto-Trade"], "N/A", "N/A", "Fusion")
        
        res = eng.execute_trade(sig, 25.0)
        print(f"Exec Result: {res}")
        
        df_new = pd.read_csv(JOURNAL_PATH)
        assert len(df_new) == 3
        last_row = df_new.iloc[-1]
        assert last_row['Strategy'] == "Auto-Vision"
        assert last_row['Amount'] == 25.0
        print("✅ Integration Verified")
        
        # 4. Test Metrics
        print("\n📊 Testing Metrics...")
        # Mock some results
        df_new.loc[0, 'Result'] = 'WIN'
        df_new.loc[0, 'PnL'] = 85.0 # 85% payout
        df_new.loc[1, 'Result'] = 'LOSS'
        
        # Re-save to simulate state change
        df_new.to_csv(JOURNAL_PATH, index=False)
        
        metrics = journal.get_metrics()
        print(f"Metrics: {metrics}")
        
        assert metrics['total'] == 3
        assert metrics['wins'] == 1
        assert metrics['losses'] == 1
        assert metrics['win_rate'] == 0.5 # 1 win / 2 completed
        assert metrics['pnl'] == 85.0
        print("✅ Metrics Verified")
        
    finally:
        # Cleanup / Restore
        if os.path.exists(JOURNAL_PATH + ".bak"):
            shutil.move(JOURNAL_PATH + ".bak", JOURNAL_PATH)
            print("\nℹ️ Restored original journal.")
        else:
            # If no backup was needed (clean slate), remove the test one
            if os.path.exists(JOURNAL_PATH):
                os.remove(JOURNAL_PATH)
                print("\nℹ️ Cleaned up test journal.")
    
    print("\n✅ Phase 25 Tests Passed!")

if __name__ == "__main__":
    test_phase25_journal()
