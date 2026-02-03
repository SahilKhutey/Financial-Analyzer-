"""
Step 14: Institutional Risk Guard.
Global Volatility & Regime Filter.
"""

import pandas as pd
from dataclasses import dataclass

@dataclass
class RiskStatus:
    is_safe: bool
    level: str # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    reason: str
    color: str # "green", "orange", "red"

class RiskEngine:
    """
    The "No-Trade" Gatekeeper.
    Enterprise Logic: Crash Detection, Liquidity Shield, Volatility Filtering.
    """
    
    def analyze(self, df: pd.DataFrame) -> RiskStatus:
        """
        Check global risk conditions.
        """
        if len(df) < 50:
            return RiskStatus(True, "LOW", "Insufficient Data", "green")
            
        # 1. Volatility Spike Check (ATR/Close)
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean().iloc[-1]
        
        last_close = close.iloc[-1]
        vol_ratio = atr_14 / last_close
        
        # 2. Crash Detection (Acceleration)
        # 2nd Derivative of Price
        velocity = close.diff()
        acceleration = velocity.diff()
        accel_val = acceleration.iloc[-1]
        
        # Heuristic: If acceleration is extremely negative, possible flash crash
        crash_threshold = -last_close * 0.02 # 2% drop acceleration per minute
        
        # 3. Liquidity Check (Volume Gaps)
        vol_sma = df['Volume'].rolling(20).mean().iloc[-1]
        current_vol = df['Volume'].iloc[-1]
        
        # LOGIC
        
        # A. Critical Stops
        if accel_val < crash_threshold:
             return RiskStatus(False, "CRITICAL", "Crash Detected (High Accel)", "red")

        if vol_ratio > 0.04: # >4% Daily Range (Extreme Vol)
            return RiskStatus(False, "CRITICAL", f"Extreme Volatility ({vol_ratio:.1%})", "red")
            
        if current_vol < vol_sma * 0.1: # 10% of avg volume
            return RiskStatus(False, "HIGH", "Low Liquidity (Dead Market)", "red")
            
        # B. Warnings
        if vol_ratio > 0.02:
            return RiskStatus(True, "MEDIUM", "Elevated Volatility", "orange")
            
        return RiskStatus(True, "LOW", "Stable Market", "green")
