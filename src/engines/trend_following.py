"""
Step 13: Trend Following Strategy Engine.
Focus: Large directional moves, macro alignment (15m - 4h).
"""

import pandas as pd
import numpy as np
from src.core.types import FinalSignal, TradeAction

class TrendFollowingEngine:
    """
    Rides the trend. Ignores the noise.
    """
    
    def analyze(self, df: pd.DataFrame, vision_out=None) -> FinalSignal:
        """
        Analyze for Trend Continuation.
        """
        if df is None:
             if vision_out:
                 signal = TradeAction.STAY_OUT
                 confidence = 0.5
                 reasons = ["Vision-Only Mode"]
                 
                 # Trend engine relies on bias
                 if vision_out.market_bias.value == "Bullish":
                     signal = TradeAction.BUY
                     confidence = 0.65
                     reasons.append("Visual Trend: Bullish")
                 elif vision_out.market_bias.value == "Bearish":
                     signal = TradeAction.SELL
                     confidence = 0.65
                     reasons.append("Visual Trend: Bearish")
                     
                 return FinalSignal(signal, confidence, reasons, vision_out.market_bias.value, "N/A", "Vision-Trend")
             return FinalSignal(TradeAction.STAY_OUT, 0.0, ["No Data"], "Neutral", "Neutral", "Unknown")

        if len(df) < 200:
             return FinalSignal(TradeAction.STAY_OUT, 0.0, ["Insufficient Data (Need 200 bars)"], "Neutral", "Neutral", "Unknown")
             
        data = df.copy()
        closes = data['Close']
        highs = data['High']
        lows = data['Low']
        
        # 1. Trend Identity (The "Golden" Filter)
        ema_50 = closes.ewm(span=50, adjust=False).mean()
        ema_200 = closes.ewm(span=200, adjust=False).mean()
        
        last_close = closes.iloc[-1]
        last_ema50 = ema_50.iloc[-1]
        last_ema200 = ema_200.iloc[-1]
        
        # Trend Status
        bull_trend = ema_50.iloc[-1] > ema_200.iloc[-1]
        bear_trend = ema_50.iloc[-1] < ema_200.iloc[-1]
        
        # 2. ADX Filter (Strength)
        # Simplified ADX calc for brevity
        def calc_adx(df, n=14):
            # ... implementation ... 
            # For simplicity in this demo engine, we'll use a Slope proxy for Strength
            # Real ADX requires Smoothed TR and DM.
            # Proxy: Slope of EMA50
            slope = (ema_50.iloc[-1] - ema_50.iloc[-5]) / 5
            strength = abs(slope) / last_close * 10000 # Normalize roughly
            return strength
            
        trend_strength = calc_adx(data)
        is_strong = trend_strength > 2.0 # Arbitrary threshold for this proxy
        
        signal = TradeAction.STAY_OUT
        confidence = 0.0
        reasons = []
        
        # 3. Structure / Entry Logic (Pullback)
        
        # BULLISH SETUP
        if bull_trend:
            reasons.append("Macro Trend: Bullish (EMA50 > EMA200)")
            
            # Check for Pullback
            # Price is close to EMA50, but above EMA200
            dist_to_50 = (last_close - last_ema50) / last_ema50
            
            if dist_to_50 > 0.02: # 2% above EMA50
                reasons.append("Price Extended (Wait for Pullback)")
            elif dist_to_50 < -0.01: # 1% below EMA50
                reasons.append("Deep Pullback (Caution: Structure Break?)")
            else:
                # In the "Golden Zone" (Near EMA50)
                if is_strong:
                     signal = TradeAction.BUY
                     confidence = 0.8
                     reasons.append("Perfect Pullback to EMA50 Value Zone")
                     reasons.append(f"Trend Strength High ({trend_strength:.1f})")
                else:
                     reasons.append("Trend Weak/Choppy (ADX Low)")
                     confidence = 0.4
                     
        # BEARISH SETUP
        elif bear_trend:
            reasons.append("Macro Trend: Bearish (EMA50 < EMA200)")
            
            dist_to_50 = (last_close - last_ema50) / last_ema50
            
            if dist_to_50 < -0.02: 
                reasons.append("Price Extended (Wait for Pullback)")
            elif dist_to_50 > 0.01:
                reasons.append("Deep Pullback (Caution)")
            else:
                if is_strong:
                     signal = TradeAction.SELL
                     confidence = 0.8
                     reasons.append("Perfect Pullback to EMA50 Value Zone")
                     reasons.append(f"Trend Strength High ({trend_strength:.1f})")
                else:
                     reasons.append("Trend Weak/Choppy (ADX Low)")
                     confidence = 0.4

        return FinalSignal(
            action=signal,
            confidence=confidence,
            reasoning=reasons,
            vision_bias="Simulated",
            ts_bias="Simulated",
            regime="TrendFollowing"
        )
