import os
import time
from src.core.logger import SystemLogger
from src.core.profiler import Profiler, LatencyStats
from src.core.config import LOGS_DIR

def test_phase27_hardening():
    print("🚀 [TEST] Phase 27: System Hardening")
    
    # 1. Test Logger
    print("📝 Testing System Logger...")
    logger = SystemLogger()
    test_msg = f"Test Log Entry {time.time()}"
    logger.info(test_msg)
    
    log_path = os.path.join(LOGS_DIR, "system.log")
    assert os.path.exists(log_path), "Log file not created"
    
    with open(log_path, 'r') as f:
        content = f.read()
        assert test_msg in content, "Log message not found in file"
    print("✅ Logger Core Verified")
    
    # 2. Test Profiler
    print("⏱️ Testing Profiler...")
    stats = LatencyStats()
    
    with Profiler("TestBlock", stats):
        time.sleep(0.1) # 100ms
        
    timings = stats.get_display_data()
    print(f"Captured Timings: {timings}")
    
    assert "TestBlock" in timings
    # Allow some jitter, but should be close to 100ms
    assert timings["TestBlock"] >= 90.0 
    print("✅ Profiler Verified")
    
    print("\n✅ Phase 27 Tests Passed!")

if __name__ == "__main__":
    test_phase27_hardening()
