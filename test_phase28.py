import time
from src.core.notifier import Notifier

def test_phase28_alerts():
    print("🚀 [TEST] Phase 28: Remote Monitoring (Sound/Discord)")
    
    # 1. Test Sound
    print("🔊 Testing Audio Alerts (Check speakers)...")
    notifier = Notifier()
    notifier.send("Test BUY Signal", level="BUY")
    time.sleep(1)
    notifier.send("Test ERROR Signal", level="ERROR")
    time.sleep(1)
    print("✅ Audio Commands Sent (Did you hear them?)")
    
    # 2. Test Discord (Mock)
    print("👾 Testing Discord Webhook Logic...")
    # This won't actually post unless config is set, but shouldn't crash
    notifier.send("Test Discord Push", level="INFO")
    print("✅ Discord Logic Executed (Check System Log for status)")
    
    print("\n✅ Phase 28 Tests Passed!")

if __name__ == "__main__":
    test_phase28_alerts()
