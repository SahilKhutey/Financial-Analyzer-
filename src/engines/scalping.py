"""
Step 10: Scalping Strategy Engine.
Focus: High-probability short-duration trades (15s - 5m).
"""

import pandas as pd
import numpy as np
from src.core.types import FinalSignal, TradeAction
from src.core.config import CONFIDENCE_THRESHOLD

class ScalpingEngine:
    """
     specialized engine for micro-trend scalping.
    """
    
    def analyze(self, df: pd.DataFrame, vision_out=None) -> FinalSignal:
        """
        Analyze price action for scalping setups.
        Supports DataFrame or Vision input.
        """
        if df is None:
             if vision_out:
                 # Vision-Only Scalping Logic
                 signal = TradeAction.STAY_OUT
                 confidence = 0.5
                 reasons = ["Vision-Only Mode"]
                 
                 if vision_out.momentum_score > 0.7:
                     signal = TradeAction.BUY
                     confidence = 0.7
                     reasons.append("Strong Visual Momentum")
                 elif vision_out.momentum_score < 0.3:
                     signal = TradeAction.SELL
                     confidence = 0.7
                     reasons.append("Weak Visual Momentum")
                     
                 return FinalSignal(signal, confidence, reasons, vision_out.market_bias.value, "N/A", "Vision")
             return FinalSignal(TradeAction.STAY_OUT, 0.0, ["No Data"], "Neutral", "Neutral", "Unknown")

        if len(df) < 50:
            return FinalSignal(TradeAction.STAY_OUT, 0.0, ["Insufficient Data"], "Neutral", "Neutral", "Unknown")
            
        # 1. Indicator Setup (Fast EMAs)
        # We calculate on the fly as feature engineer might be geared for Swing
        closes = df['Close']
        ema_9 = closes.ewm(span=9, adjust=False).mean()
        ema_20 = closes.ewm(span=20, adjust=False).mean()
        
        last_close = closes.iloc[-1]
        last_ema9 = ema_9.iloc[-1]
        last_ema20 = ema_20.iloc[-1]
        
        # 2. Logic: Trend & Pullback
        # Bullish: Price > EMA20, Price pulls back to near EMA9
        # Bearish: Price < EMA20, Price pulls back to near EMA9
        
        signal = TradeAction.STAY_OUT
        confidence = 0.0
        reasons = []
        
        # Trend Filter
        trend_bull = last_ema9 > last_ema20
        trend_bear = last_ema9 < last_ema20
        
        # Distance to EMA9 (The "Pullback" Zone)
        dist_pct = abs(last_close - last_ema9) / last_ema9
        is_pullback = dist_pct < 0.001 # Extremely tight for scalping simulation
        
        # Momentum (RSI) - Check if calculated
        rsi_val = 50
        if 'rsi' in df.columns:
            rsi_val = df['rsi'].iloc[-1]
        elif len(closes) > 15:
             # Quick Calc
             delta = closes.diff()
             gain = (delta.where(delta > 0, 0)).rolling(14).mean()
             loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
             rs = gain/loss
             rsi_val = (100 - (100/(1+rs))).iloc[-1]

        # 3. Decision
        if trend_bull:
            if last_close > last_ema20:
                if rsi_val > 50 and rsi_val < 70: # Momentum supporting but not overbought
                    signal = TradeAction.BUY
                    confidence = 0.75
                    reasons.append("Bullish Micro-Trend (EMA9 > EMA20)")
                    reasons.append(f"Momentum Active (RSI {rsi_val:.0f})")
                    if is_pullback:
                        confidence += 0.1
                        reasons.append("Perfect EMA9 Pullback")
        
        elif trend_bear:
            if last_close < last_ema20:
                if rsi_val < 50 and rsi_val > 30:
                    signal = TradeAction.SELL
                    confidence = 0.75
                    reasons.append("Bearish Micro-Trend (EMA9 < EMA20)")
                    reasons.append(f"Momentum Active (RSI {rsi_val:.0f})")
                    if is_pullback:
                        confidence += 0.1
                        reasons.append("Perfect EMA9 Pullback")
                        
        # 4. Volatility Gate (Scalping specific)
        # Using simple body size analysis
        last_open = df['Open'].iloc[-1]
        body_size = abs(last_close - last_open)
        avg_body = (df['Close'] - df['Open']).abs().rolling(10).mean().iloc[-1]
        
        if body_size > avg_body * 3:
            signal = TradeAction.STAY_OUT
            reasons.append("Volatility Spike Detected (Candle > 3x Avg)")
            confidence = 0.0
            
        return FinalSignal(
            action=signal,
            confidence=confidence,
            reasoning=reasons,
            vision_bias="N/A (Fast)",
            ts_bias="N/A (Fast)",
            regime="Scalping"
        )
