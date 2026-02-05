"""
Step 28: Notification Engine.
Handles local audio alerts and remote Discord push notifications.
"""
import sys
import threading
import requests
import time
from src.core.config import ENABLE_SOUND, DISCORD_WEBHOOK_URL
from src.core.logger import SystemLogger

# Conditional Import for Windows Sound
try:
    import winsound
except ImportError:
    winsound = None

class Notifier:
    """
    Singleton Notification Manager.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Notifier, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
        
    def _initialize(self):
        self.logger = SystemLogger()
        
    def _play_sound(self, frequency, duration):
        if winsound and ENABLE_SOUND:
            try:
                winsound.Beep(frequency, duration)
            except Exception:
                pass

    def _sound_thread(self, alert_type: str):
        """Run sound in thread to not block main loop."""
        if alert_type == "BUY":
            self._play_sound(500, 150)
            time.sleep(0.1)
            self._play_sound(1000, 300) # Ascending
        elif alert_type == "SELL":
            self._play_sound(1000, 150)
            time.sleep(0.1)
            self._play_sound(500, 300) # Descending
        elif alert_type == "ERROR":
            self._play_sound(200, 800)
            
    def _discord_thread(self, msg: str):
        """Send Discord Webhook in thread."""
        if DISCORD_WEBHOOK_URL:
            try:
                payload = {"content": msg}
                requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
            except Exception as e:
                self.logger.error(f"Discord Webhook Failed: {e}")

    def send(self, msg: str, level: str = "INFO"):
        """
        Trigger an alert.
        Level: INFO, BUY, SELL, ERROR
        """
        # 1. Log IT
        if level == "ERROR":
            self.logger.error(msg)
        else:
            self.logger.info(f"ALERT: {msg}")
            
        # 2. Sound (Threaded)
        if ENABLE_SOUND:
             t = threading.Thread(target=self._sound_thread, args=(level,))
             t.daemon = True
             t.start()
             
        # 3. Discord (Threaded)
        if DISCORD_WEBHOOK_URL:
            fancy_msg = msg
            if level == "BUY": fancy_msg = f"🟢 **BUY SIGNAL**: {msg}"
            elif level == "SELL": fancy_msg = f"🔴 **SELL SIGNAL**: {msg}"
            elif level == "ERROR": fancy_msg = f"⚠️ **CRITICAL ERROR**: {msg}"
            
            t = threading.Thread(target=self._discord_thread, args=(fancy_msg,))
            t.daemon = True
            t.start()
