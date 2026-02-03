"""
Binomo Auto-Signal Engine.
Specialized for High-Frequency Binary Options (1m/5m).

Logic:
- Signal: Probabilistic Scoring (Heuristic Classifier).
- Filter: Strict RSI & Volatility Gates.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class BinomoSignal:
    action: str  # "BUY", "SELL", "STAY_OUT"
    confidence: float # 0.0 to 1.0
    expiry: str # "2-3 min"
    reasoning: List[str]
    is_safe: bool

class BinomoEngine:
    def __init__(self):
        self.min_volatility = 0.0004
        self.rsi_upper = 70
        self.rsi_lower = 30
        
    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def risk_filter(self, row: pd.Series) -> tuple[bool, str]:
        """
        Step 3: Binomo-Safe Risk Filter
        """
        # RSI Check
        rsi = row.get('rsi', 50)
        if rsi > self.rsi_upper or rsi < self.rsi_lower:
            return False, f"Risk: RSI Extreme ({rsi:.1f})"
            
        # Volatility Check (Normalized Range)
        # Using High-Low / Open as proxy if 'volatility' not pre-calc
        vol = row.get('volatility', 0)
        if vol == 0:
             # Fallback calc
             vol = (row['High'] - row['Low']) / row['Open']
        
        if vol < self.min_volatility:
            return False, f"Risk: Low Volatility ({vol:.5f})"
            
        return True, "Safe"

    def generate_signal(self, row: pd.Series) -> dict:
        """
        Step 2: Signal Engine (Simulated Probabilistic Classifier)
        Returns dict with signal and confidence.
        """
        score = 0.5 # Neutral
        
        # 1. Trend (EMA-like Logic)
        # Assuming we have some trend info or calculating generic momentum
        close = row['Close']
        open_p = row['Open']
        
        # Simple Momentum
        if close > open_p: score += 0.1
        else: score -= 0.1
        
        # RSI Influence (Mean Reversion pressure vs Momentum)
        rsi = row.get('rsi', 50)
        if rsi > 50: score += 0.05
        else: score -= 0.05
        
        # Aggressive Momentum Boost
        if abs(close - open_p) / open_p > 0.001:
            if close > open_p: score += 0.15
            else: score -= 0.15
            
        # Normalize to 0-1
        prob_buy = min(0.99, max(0.01, score))
        prob_sell = 1.0 - prob_buy
        
        threshold = 0.60
        
        if prob_buy > threshold:
            return {"signal": "BUY", "confidence": prob_buy}
        elif prob_sell > threshold:
            return {"signal": "SELL", "confidence": prob_sell}
        else:
            return {"signal": "STAY_OUT", "confidence": max(prob_buy, prob_sell)}

    def analyze(self, df: pd.DataFrame, vision_out=None) -> BinomoSignal:
        """
        Main Analysis pipeline.
        """
        if df is None or df.empty:
            return BinomoSignal("STAY_OUT", 0.0, "N/A", ["No Data for Logic"], False)
            
        # Ensure features
        if 'rsi' not in df.columns:
            df['rsi'] = self._calculate_rsi(df['Close'])
            
        latest = df.iloc[-1]
        
        # 1. Risk Filter
        safe, reason = self.risk_filter(latest)
        if not safe:
            return BinomoSignal("STAY_OUT", 0.0, "N/A", [reason], False)
            
        # 2. Generate Signal
        raw_sig = self.generate_signal(latest)
        
        # 3. Vision Confirmation (Optional Boost)
        conf = raw_sig['confidence']
        reasons = [f"Model Prob: {conf:.2f}"]
        
        if vision_out:
            reasons.append(f"Vision Bias: {vision_out.market_bias.value}")
            if raw_sig['signal'] == "BUY" and vision_out.market_bias.value == "BULLISH":
                conf = min(0.99, conf + 0.1)
            elif raw_sig['signal'] == "SELL" and vision_out.market_bias.value == "BEARISH":
                conf = min(0.99, conf + 0.1)
                
        return BinomoSignal(
            action=raw_sig['signal'],
            confidence=conf,
            expiry="2-3 min",
            reasoning=reasons,
            is_safe=True
        )
