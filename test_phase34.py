from src.interfaces.telegram_bot import TelegramBot

def test_phase34_bot():
    print("🚀 [TEST] Phase 34: Remote Command Interface")
    
    # 1. Test Bot Init
    print("Initializing Bot...")
    bot = TelegramBot()
    assert bot.token == "SIMULATION_MODE"
    print("✅ Bot Initialized")
    
    # 2. Test Command Injection
    print("Testing Command Injection...")
    bot.inject_command("/status")
    bot.inject_command("/stop")
    
    # 3. Test Polling
    print("Testing Polling...")
    
    cmd1 = bot.poll()
    print(f"Received: {cmd1.command}")
    assert cmd1.command == "/status"
    
    cmd2 = bot.poll()
    print(f"Received: {cmd2.command}")
    assert cmd2.command == "/stop"
    
    cmd3 = bot.poll()
    print(f"Received: {cmd3}")
    assert cmd3 is None
    
    print("✅ Polling Verified")
    
    # 4. Test Sending
    print("Testing Send Message...")
    bot.send_message("Test Message")
    print("✅ Send Logic Verified (Simulated)")

    print("\n✅ Phase 34 Tests Passed!")

if __name__ == "__main__":
    test_phase34_bot()
