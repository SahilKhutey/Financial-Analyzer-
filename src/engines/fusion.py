"""
Step 5: Signal Fusion Brain (Grand Fusion).
Aggregates Independent Signals into a Cohesive Institutional Trade Decision.

Weights:
- Enterprise ML (XGB+LSTM): 35%
- Deep Analytical (Inst.): 25%
- Mathematical (Stoch): 15%
- Vision (Legacy): 15%
- TimeSeries (Legacy): 10%
"""

from src.core.types import FinalSignal, VisionOutput, TSOutput, RegimeOutput, TradeAction, MarketBias
from src.core.gates import SafetyGates
import numpy as np

class SignalFusion:
    """
    The Meta-Decision Engine.
    Combines 5+ independent institutional models.
    """
    
    def __init__(self):
        # Grand Ensemble Weights
        self.w_ml = 0.35
        self.w_deep = 0.25
        self.w_math = 0.15
        self.w_vision = 0.15
        self.w_ts = 0.10
        
    def _signal_to_score(self, signal: FinalSignal) -> float:
        """Convert FinalSignal to -1 to 1 float."""
        if signal.action == TradeAction.BUY:
            return signal.confidence
        elif signal.action == TradeAction.SELL:
            return -signal.confidence
        else:
            return 0.0

    def fuse(self, 
             vision: VisionOutput, 
             ts: Optional[TSOutput] = None, 
             regime: Optional[RegimeOutput] = None,
             ml_sig: FinalSignal = None,
             deep_sig: FinalSignal = None,
             math_sig: FinalSignal = None) -> FinalSignal:
        """
        Produce the Final Grand Signal.
        Auto-adapts weights if data-driven signals are missing (Vision-Only Mode).
        """
        # 1. Component Scores (-1 to 1)
        
        # Vision Score
        v_score = 0.0
        if vision:
            v_score = 1.0 if vision.market_bias == MarketBias.BULLISH else (-1.0 if vision.market_bias == MarketBias.BEARISH else 0.0)
            v_score *= vision.breakout_prob # Confidence scaling
        
        # TS Score
        ts_score = 0.0
        if ts:
            ts_score = (ts.bullish_prob - ts.bearish_prob)
            
        # New Engine Scores
        ml_score = self._signal_to_score(ml_sig) if ml_sig else 0.0
        deep_score = self._signal_to_score(deep_sig) if deep_sig else 0.0
        math_score = self._signal_to_score(math_sig) if math_sig else 0.0
        
        # 2. Dynamic Weighting Logic
        # If we lack data-driven signals (TS, ML, Deep, Math), we are in "Vision Only" mode.
        
        current_weights = {
            'ml': self.w_ml if ml_sig else 0.0,
            'deep': self.w_deep if deep_sig else 0.0,
            'math': self.w_math if math_sig else 0.0,
            'ts': self.w_ts if ts else 0.0,
            'vision': self.w_vision
        }
        
        total_weight = sum(current_weights.values())
        
        if total_weight == 0:
            return FinalSignal(TradeAction.STAY_OUT, 0.0, ["No Signals Available"], "N/A", "N/A", "N/A")
            
        # Normalize weights to sum to 1.0
        norm_w = {k: v / total_weight for k, v in current_weights.items()}
        
        # 3. Weighted Sum
        raw_signal = (
            (ml_score * norm_w['ml']) +
            (deep_score * norm_w['deep']) +
            (math_score * norm_w['math']) +
            (v_score * norm_w['vision']) +
            (ts_score * norm_w['ts'])
        )
        
        # 4. Decision Logic & Confluence Check
        action = TradeAction.STAY_OUT
        confidence = abs(raw_signal)
        
        # Thresholds
        BUY_THRESH = 0.25
        SELL_THRESH = -0.25
        
        # Veto Logic: If ML (Strongest) is strongly opposing the weighted sum, reduce confidence
        if ml_sig and ml_sig.confidence > 0.6:
            if (ml_score > 0 and raw_signal < 0) or (ml_score < 0 and raw_signal > 0):
                # Strong disagreement -> Kill signal
                raw_signal = 0.0
                confidence = 0.0
        
        if raw_signal > BUY_THRESH:
            action = TradeAction.BUY
        elif raw_signal < SELL_THRESH:
            action = TradeAction.SELL
            
        # 5. Construct Final Signal
        reasons = []
        mode_label = "Grand Fusion"
        
        if norm_w['vision'] > 0.9: 
            mode_label = "Vision Only"
            reasons.append("⚠️ Vision Only Mode")
            
        reasons.append(f"Score: {raw_signal:.2f}")
        if ml_sig: reasons.append(f"ML: {ml_score:.2f}")
        if norm_w['vision'] > 0.5: reasons.append(f"Visual: {v_score:.2f}")
        
        regime_state = regime.state if regime else "Unknown"
        
        intermediate_signal = FinalSignal(
            action=action,
            confidence=round(confidence, 2),
            reasoning=reasons,
            vision_bias=vision.market_bias.value if vision else "N/A",
            ts_bias=f"{mode_label}:{raw_signal:.2f}",
            regime=regime_state
        )
        
        # 6. Apply Safety Gates (Final Filter)
        # Note: Gates might block if Regime is missing/Unknown. We might need to relax Gates for Vision Only.
        if regime:
             final_signal = SafetyGates.validate(intermediate_signal, vision, ts, regime)
        else:
             final_signal = intermediate_signal # Skip gates if no regime data (Vision Only Risk is high!)
        
        return final_signal
