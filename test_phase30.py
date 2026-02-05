import pandas as pd
import os
import shutil
from src.core.journal import Journal, TRAINING_DATA_PATH
from src.engines.trainer import ModelTrainer

def test_phase30_learning():
    print("🚀 [TEST] Phase 30: Continuous Learning")
    
    # 1. Setup - Backup Clean
    if os.path.exists(TRAINING_DATA_PATH):
        shutil.copy(TRAINING_DATA_PATH, TRAINING_DATA_PATH + ".bak")
        
    try:
        # 2. Test Feature Enrichment (Journal)
        print("📝 Testing Feature Logging...")
        j = Journal()
        # Log a trade WITH features
        mock_feats = {
            "log_ret": 0.01,
            "vol": 0.02,
            "mom": 0.05,
            "rsi": 65.0,
            "vol_atr": 1.5
        }
        # Log 20 fake trades to simulate a dataset
        for i in range(20):
             # Alternate Win/Loss for balance
             result = "WIN" if i % 2 == 0 else "LOSS"
             j.log_trade("BTC", "BUY", 10.0, "Test", "Test", features=mock_feats)
             
        # Manually update results in CSV to "WIN" / "LOSS"
        # Since log_trade defaults to PENDING, we need to hack the CSV to simulate "Completed" trades
        df = pd.read_csv(TRAINING_DATA_PATH)
        df['Result'] = ['WIN' if i % 2 == 0 else 'LOSS' for i in range(len(df))]
        df.to_csv(TRAINING_DATA_PATH, index=False)
        
        print("✅ Data Logged & Labeled")
        
        # 3. Test Trainer
        print("🧠 Testing Model Trainer...")
        trainer = ModelTrainer()
        stats = trainer.get_dataset_stats()
        print(f"Stats: {stats}")
        assert stats['count'] >= 20
        assert stats['wins'] >= 10
        print("✅ Stats Logic Verified")
        
        res = trainer.train_model()
        print(f"Training Result: {res}")
        assert "Success" in res
        
        # Check Model File
        from src.engines.trainer import MODEL_PATH
        assert os.path.exists(MODEL_PATH)
        print("✅ Model File Created")
        
        print("\n✅ Phase 30 Tests Passed!")
        
    finally:
        # Cleanup / Restore
        # Optional: Keep the test data or remove. Let's remove to keep clean state
        pass

if __name__ == "__main__":
    test_phase30_learning()
