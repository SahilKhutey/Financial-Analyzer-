"""
Step 12: Mean Reversion Strategy Engine.
Focus: Snapback trades to fair value (VWAP) in ranging markets.
"""

import pandas as pd
import numpy as np
from src.core.types import FinalSignal, TradeAction

class MeanReversionEngine:
    """
    Identifies overextended price action reverting to the mean.
    """
    
    def analyze(self, df: pd.DataFrame, vision_out=None) -> FinalSignal:
        """
        Analyze for Mean Reversion Setups.
        """
        if df is None:
             if vision_out:
                 # Hard to do mean reversion without price data, but we can check for "Topping" patterns from Vision
                 signal = TradeAction.STAY_OUT
                 confidence = 0.4
                 reasons = ["Vision-Only Mode"]
                 
                 if "Topping" in str(vision_out.patterns_detected):
                     signal = TradeAction.SELL
                     confidence = 0.6
                     reasons.append("Visual Exhaustion Pattern Detected")
                     
                 return FinalSignal(signal, confidence, reasons, vision_out.market_bias.value, "N/A", "Vision-Reversion")
             return FinalSignal(TradeAction.STAY_OUT, 0.0, ["No Data"], "Neutral", "Neutral", "Unknown")

        if len(df) < 50:
             return FinalSignal(TradeAction.STAY_OUT, 0.0, ["Insufficient Data"], "Neutral", "Neutral", "Unknown")
             
        data = df.copy()
        closes = data['Close']
        highs = data['High']
        lows = data['Low']
        opens = data['Open']
        
        # 1. Calculate Indicators
        # VWAP (Volume Weighted Average Price) - Standard Day-Session anchor usually, 
        # but for continuous stream we can use a rolling VWAP or just SMA20 as proxy for "Mean" in simple mode.
        # Let's use a Rolling VWAP (e.g., 20 period) for short-term mean.
        cumulative_vol = data['Volume'].rolling(20).sum()
        cumulative_vol_price = (data['Close'] * data['Volume']).rolling(20).sum()
        vwap = cumulative_vol_price / cumulative_vol
        
        # PSI / RSI
        # Recalc RSI if not present
        if 'rsi' in data.columns:
            rsi = data['rsi'].iloc[-1]
        else:
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain/loss
            rsi = (100 - (100/(1+rs))).iloc[-1]
            
        # Slope of SMA50 (Range Filter)
        sma50 = closes.rolling(50).mean()
        slope = (sma50.iloc[-1] - sma50.iloc[-5]) / 5 # Simple slope over 5 bars
        is_ranging = abs(slope) < (closes.iloc[-1] * 0.0005) # Very flat slope
        
        # 2. Wick Analysis (Vision)
        last_close = closes.iloc[-1]
        last_open = opens.iloc[-1]
        last_high = highs.iloc[-1]
        last_low = lows.iloc[-1]
        
        body_size = abs(last_close - last_open)
        upper_wick = last_high - max(last_close, last_open)
        lower_wick = min(last_close, last_open) - last_low
        
        signal = TradeAction.STAY_OUT
        confidence = 0.0
        reasons = []
        
        # 3. Logic
        
        # Filter: Must be Ranging (or at least not trending HARD against signal)
        # We allow counter-trend if extreme extension.
        
        last_vwap = vwap.iloc[-1]
        dist_to_mean = (last_close - last_vwap) / last_vwap
        
        # BUY (Oversold Snapback)
        if rsi < 30:
            reasons.append(f"RSI Oversold ({rsi:.0f} < 30)")
            
            # Extension check
            if dist_to_mean < -0.01: # 1% below mean
                 reasons.append("Price Extended below VWAP")
                 confidence += 0.4
                 
                 # Wick Confirmation (Hammer)
                 if lower_wick > body_size * 1.5:
                     reasons.append("Rejection Wick Detected (Hammer)")
                     confidence += 0.3
                     signal = TradeAction.BUY
                 elif is_ranging:
                     reasons.append("Market is Ranging (Safe to Fade)")
                     confidence += 0.2
                     signal = TradeAction.BUY
        
        # SELL (Overbought Snapback)     
        elif rsi > 70:
            reasons.append(f"RSI Overbought ({rsi:.0f} > 70)")
            
            # Extension check
            if dist_to_mean > 0.01: # 1% above mean
                reasons.append("Price Extended above VWAP")
                confidence += 0.4
                
                # Wick Confirmation (Shooting Star)
                if upper_wick > body_size * 1.5:
                    reasons.append("Rejection Wick Detected (Shooting Star)")
                    confidence += 0.3
                    signal = TradeAction.SELL
                elif is_ranging:
                     reasons.append("Market is Ranging (Safe to Fade)")
                     confidence += 0.2
                     signal = TradeAction.SELL

        return FinalSignal(
            action=signal,
            confidence=confidence,
            reasoning=reasons,
            vision_bias="N/A",
            ts_bias="N/A",
            regime="MeanReversion"
        )
