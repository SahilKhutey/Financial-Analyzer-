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
from src.core.types import FinalSignal, TradeAction, VisionOutput
from src.core.journal import Journal
from src.core.notifier import Notifier
from src.core.money_manager import MoneyManager

@dataclass
class BinomoSignal:
    action: str  # "BUY", "SELL", "STAY_OUT"
    confidence: float # 0.0 to 1.0
    expiry: str # "1 min", "2-3 min", "5 min"
    reasoning: List[str]
    is_safe: bool

class BinomoEngine:
    def __init__(self):
        self.min_volatility = 0.0004
        self.rsi_upper = 70
        self.rsi_lower = 30
        self.journal = Journal()
        self.notifier = Notifier()
        self.money_manager = MoneyManager()
        
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
        vol = row.get('volatility', 0)
        if vol == 0:
             vol = (row['High'] - row['Low']) / row['Open']
        
        if vol < self.min_volatility:
            return False, f"Risk: Low Volatility ({vol:.5f})"
            
        return True, "Safe"

    def determine_expiry(self, volatility: float) -> str:
        """
        Dynamic Expiry based on Market Velocity.
        High Vol -> Fast Expiry (1 min).
        Low Vol -> Slow Expiry (5 min).
        """
        if volatility > 0.002: # High movement
            return "1 min (Fast)"
        elif volatility < 0.0008: # Slow crawl
            return "5 min (Trend)"
        else:
            return "2-3 min (Normal)"

    def analyze(self, df: pd.DataFrame, vision_out: Optional[VisionOutput] = None, fusion_sig: Optional[FinalSignal] = None) -> BinomoSignal:
        """
        Main Analysis pipeline.
        Integrates "Grand Fusion" signals for institutional accuracy.
        """
        if df is None or df.empty:
            return BinomoSignal("STAY_OUT", 0.0, "N/A", ["No Data for Logic"], False)
            
        # Ensure features
        if 'rsi' not in df.columns:
            df['rsi'] = self._calculate_rsi(df['Close'])
            
        latest = df.iloc[-1]
        
        # 0. Calc Volatility for Expiry
        vol = latest.get('volatility', (latest['High'] - latest['Low']) / latest['Open'])
        expiry = self.determine_expiry(vol)
        
        # 1. Risk Filter
        safe, reason = self.risk_filter(latest)
        if not safe:
            return BinomoSignal("STAY_OUT", 0.0, "N/A", [reason], False)
            
        # 2. DECISION LOGIC
        
        # A. Institutional Override (Fusion)
        # If we have a High Confidence Fusion Signal, we use it directly.
        if fusion_sig and fusion_sig.action != TradeAction.STAY_OUT and fusion_sig.confidence > 0.7:
             return BinomoSignal(
                action=fusion_sig.action.value,
                confidence=fusion_sig.confidence,
                expiry=expiry,
                reasoning=["🔥 Institutional Override"] + fusion_sig.reasoning,
                is_safe=True
            )
            
        # B. Fallback: Local Heuristic (If Fusion is weak/missing)
        score = 0.5
        reasons = []
        
        # Trend
        if latest['Close'] > latest['Open']: 
            score += 0.1
            reasons.append("Candle Green")
        else: 
            score -= 0.1
            reasons.append("Candle Red")
            
        # RSI
        rsi = latest['rsi']
        if rsi > 55: score += 0.05
        elif rsi < 45: score -= 0.05
        
        # Normalize
        prob_buy = min(0.99, max(0.01, score))
        prob_sell = 1.0 - prob_buy
        
        final_action = "STAY_OUT"
        final_conf = 0.0
        
        if prob_buy > 0.65:
            final_action = "BUY"
            final_conf = prob_buy
        elif prob_sell > 0.65:
            final_action = "SELL"
            final_conf = prob_sell
            
        # C. Vision Boost
        if vision_out:
            if final_action == "BUY" and vision_out.market_bias.value == "BULLISH":
                final_conf = min(0.99, final_conf + 0.1)
                reasons.append(f"Vision Confirmed")
            elif final_action == "SELL" and vision_out.market_bias.value == "BEARISH":
                final_conf = min(0.99, final_conf + 0.1)
                reasons.append(f"Vision Confirmed")
                
        return BinomoSignal(
            action=final_action,
            confidence=final_conf,
            expiry=expiry,
            reasoning=reasons,
            is_safe=True
        )

    def execute_trade(self, signal: FinalSignal, amount: float = 10.0) -> str:
        """
        Mock Execution Bridge.
        In production, this would call the Binomo Private API / Selenium Driver.
        """
        if signal.action == TradeAction.STAY_OUT:
            return "No Trade"
            
        # 1. Money Management
        final_amount = self.money_manager.get_trade_amount(signal.confidence)
        
        if final_amount <= 0:
            return "Risk Gate: Trading Stopped (Daily Loss Limit)"
            
        # Log Logic
        print(f"⚡ EXECUTING BINOMO TRADE: {signal.action.value} | Amt: ${final_amount}")
        
        # Journal Logs
        self.journal.log_trade(
            asset="BTC/Crypto", # Default/Generic for now
            action=signal.action.value,
            amount=final_amount,
            strategy="Auto-Vision",
            reason=str(signal.reasoning)
        )
        
        # Alert
        self.notifier.send(f"Executed {signal.action.value} on BTC for ${final_amount}", level=signal.action.value)
        
        return f"Placed {signal.action.value} for ${final_amount} (Simulated)"
