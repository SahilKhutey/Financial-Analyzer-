import os
import shutil
import numpy as np
from src.engines.yolo_logic import YOLOEngine
from src.engines.binomo import BinomoEngine
from src.core.types import FinalSignal, TradeAction

def test_phase24_execution():
    print("🚀 [TEST] Phase 24: Execution & Data")
    
    # 1. Setup
    yolo = YOLOEngine()
    binomo = BinomoEngine()
    
    # 2. Test Dataset Recorder
    print("\n📸 Testing Image Saver...")
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Use a temp prefix to distinguish
    path = yolo.save_snapshot(dummy_img, prefix="test_capture")
    
    print(f"Saved to: {path}")
    assert os.path.exists(path), "File was not created!"
    assert "test_capture" in path
    
    # Clean up test file
    try:
        os.remove(path)
        print("✅ File Save Verified (and cleaned up)")
    except Exception as e:
        print(f"⚠️ Cleanup failed: {e}")
        
    # 3. Test Execution Bridge
    print("\n⚡ Testing Execution Bridge...")
    mock_signal = FinalSignal(TradeAction.BUY, 0.9, ["Mock Reason"], "N/A", "N/A", "Test")
    
    res = binomo.execute_trade(mock_signal, amount=50.0)
    print(f"Response: {res}")
    
    assert "Placed BUY for $50.0" in res
    print("✅ Trade Logic Verified")
    
    print("\n✅ Phase 24 Tests Passed!")

if __name__ == "__main__":
    test_phase24_execution()
