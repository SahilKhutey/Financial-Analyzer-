"""
Step 11: Breakout Strategy Engine.
Focus: Momentum Breakouts from Consolidation.
"""

import pandas as pd
import numpy as np
from src.core.types import FinalSignal, TradeAction

class BreakoutEngine:
    """
    Detects volatility expansion (Breakouts) after compression (Squeezes).
    """
    
    def analyze(self, df: pd.DataFrame, vision_out=None) -> FinalSignal:
        """
        Analyze for Breakout Setups.
        """
        if df is None:
             if vision_out:
                 signal = TradeAction.STAY_OUT
                 confidence = 0.5
                 reasons = ["Vision-Only Mode"]
                 
                 if vision_out.breakout_prob > 0.6:
                     signal = TradeAction.BUY if vision_out.market_bias.value == "Bullish" else TradeAction.SELL
                     confidence = vision_out.breakout_prob
                     reasons.append(f"Visual Breakout Detected (Prob: {vision_out.breakout_prob:.2f})")
                     
                 return FinalSignal(signal, confidence, reasons, vision_out.market_bias.value, "N/A", "Vision-Breakout")
             return FinalSignal(TradeAction.STAY_OUT, 0.0, ["No Data"], "Neutral", "Neutral", "Unknown")

        if len(df) < 50:
            return FinalSignal(TradeAction.STAY_OUT, 0.0, ["Insufficient Data"], "Neutral", "Neutral", "Unknown")
            
        data = df.copy()
        
        # 1. Bollinger Bands & Bandwidth
        # Window 20, Std 2
        sma = data['Close'].rolling(20).mean()
        std = data['Close'].rolling(20).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        
        bandwidth = (upper - lower) / sma
        
        # 2. Volume Profile
        vol_sma = data['Volume'].rolling(20).mean()
        
        # Current Candle Stats
        last_close = data['Close'].iloc[-1]
        last_open = data['Open'].iloc[-1]
        last_high = data['High'].iloc[-1]
        last_vol = data['Volume'].iloc[-1]
        last_vol_avg = vol_sma.iloc[-1]
        
        last_upper = upper.iloc[-1]
        last_lower = lower.iloc[-1]
        
        # Consolidation Check (Squeeze)
        # We look at the last 5 candles to see if bandwidth was low relative to recent history
        # Simple heuristic: Bandwidth < 10th percentile of last 50 bars? 
        # Or just "narrow" threshold. Let's use percentile for adaptiveness.
        
        recent_bw = bandwidth.iloc[-50:]
        low_vol_threshold = recent_bw.quantile(0.20) # Bottom 20% of width = Squeeze Zone
        current_bw = bandwidth.iloc[-1]
        is_squeezing = current_bw <= low_vol_threshold
        
        # Or was squeezing recently? (Breakout happens as squeeze expands)
        was_squeezing = bandwidth.iloc[-5] <= low_vol_threshold
        
        signal = TradeAction.STAY_OUT
        confidence = 0.0
        reasons = []
        
        # 3. Logic
        
        # Breakout UP
        if last_close > last_upper:
            reasons.append("Price closed above Upper Bollinger Band")
            
            # Volume Confirmation
            if last_vol > last_vol_avg * 1.5:
                confidence += 0.5
                reasons.append(f"Volume Surge ({last_vol/last_vol_avg:.1f}x avg)")
                signal = TradeAction.BUY
            else:
                 reasons.append("Low Volume (Fakeout Warn)")
                 confidence += 0.2 # Weak signal
            
            # Squeeze Confirmation (The "Pop")
            if was_squeezing or is_squeezing:
                confidence += 0.3
                reasons.append("Coming out of Consolidation (Squeeze)")
                
            # Wick Logic (Rejection check)
            # If upper wick is huge, it's a shooting star (Rejection)
            upper_wick = last_high - max(last_close, last_open)
            body = abs(last_close - last_open)
            if upper_wick > body:
                signal = TradeAction.STAY_OUT
                reasons.append("Wick Rejection Detected (Shooting Star)")
                confidence = 0.0

        # Breakout DOWN
        elif last_close < last_lower:
            reasons.append("Price closed below Lower Bollinger Band")
            
            if last_vol > last_vol_avg * 1.5:
                confidence += 0.5
                reasons.append(f"Volume Surge ({last_vol/last_vol_avg:.1f}x avg)")
                signal = TradeAction.SELL
            else:
                 reasons.append("Low Volume (Fakeout Warn)")
                 confidence += 0.2
                 
            if was_squeezing or is_squeezing:
                confidence += 0.3
                reasons.append("Coming out of Consolidation (Squeeze)")
                
            # Wick Logic
            lower_wick = min(last_close, last_open) - data['Low'].iloc[-1]
            body = abs(last_close - last_open)
            if lower_wick > body:
                signal = TradeAction.STAY_OUT
                reasons.append("Wick Rejection Detected (Hammer)")
                confidence = 0.0
        
        # 4. Final Threshold
        if signal != TradeAction.STAY_OUT and confidence < 0.6:
            signal = TradeAction.STAY_OUT
            reasons.append(f"Confidence too low ({confidence:.2f} < 0.6)")

        return FinalSignal(
            action=signal,
            confidence=confidence,
            reasoning=reasons,
            vision_bias="N/A",
            ts_bias="N/A",
            regime="Breakout"
        )
