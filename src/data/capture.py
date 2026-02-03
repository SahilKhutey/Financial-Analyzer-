"""
Step 15: Screen Capture Module.
input: OS Screen/Window.
output: PIL Image for Vision Engine.
"""

import mss
import mss.tools
import numpy as np
from PIL import Image
import threading
import time

class ScreenCapture:
    """
    Handles OS-level screen capture.
    """
    
    def __init__(self):
        self.sct = mss.mss()
        self.monitors = self.sct.monitors # List of monitors
        
    def get_monitors(self):
        """
        Returns list of available displays.
        """
        # mss.monitors[0] is 'All', [1] is Primary, etc.
        # We return a dict or list for UI
        return self.monitors
    
    def get_monitor_names(self):
        return [f"Monitor {i}: {m['width']}x{m['height']}" for i, m in enumerate(self.monitors) if i > 0]

        
    def capture(self, monitor_idx: int = 1) -> Image.Image:
        """
        Capture specific monitor.
        Args:
            monitor_idx: Index of monitor (0=All, 1=Primary, etc.)
        """
        try:
            if monitor_idx >= len(self.monitors):
                monitor_idx = 1 # Fallback to primary
                
            monitor = self.monitors[monitor_idx]
            
            # Capture
            sct_img = self.sct.grab(monitor)
            
            # Convert to PIL
            return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            
        except Exception as e:
            print(f"Capture Error: {e}")
            return None

    def capture_region(self, top, left, width, height) -> Image.Image:
        """
        Capture specific region (e.g., from a selected window coordinates).
        """
        bbox = {"top": top, "left": left, "width": width, "height": height}
        try:
            sct_img = self.sct.grab(bbox)
            return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        except Exception as e:
            return None
