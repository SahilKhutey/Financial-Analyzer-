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
             ts: TSOutput, 
             regime: RegimeOutput,
             ml_sig: FinalSignal = None,
             deep_sig: FinalSignal = None,
             math_sig: FinalSignal = None) -> FinalSignal:
        """
        Produce the Final Grand Signal.
        """
        # 1. Component Scores (-1 to 1)
        
        # Legacy Scores
        v_score = 1.0 if vision.market_bias == MarketBias.BULLISH else (-1.0 if vision.market_bias == MarketBias.BEARISH else 0.0)
        v_score *= vision.breakout_prob # Confidence scaling
        
        ts_score = (ts.bullish_prob - ts.bearish_prob)
        
        # New Engine Scores
        ml_score = self._signal_to_score(ml_sig) if ml_sig else 0.0
        deep_score = self._signal_to_score(deep_sig) if deep_sig else 0.0
        math_score = self._signal_to_score(math_sig) if math_sig else 0.0
        
        # 2. Weighted Sum
        # Normalize weights if inputs are missing? 
        # For now, we assume if missing, they contribute 0 (neutral), effectively lowering confidence.
        
        raw_signal = (
            (ml_score * self.w_ml) +
            (deep_score * self.w_deep) +
            (math_score * self.w_math) +
            (v_score * self.w_vision) +
            (ts_score * self.w_ts)
        )
        
        # 3. Decision Logic & Confluence Check
        action = TradeAction.STAY_OUT
        confidence = abs(raw_signal)
        
        # Thresholds
        BUY_THRESH = 0.25
        SELL_THRESH = -0.25
        
        # Veto Logic: If ML (Strongest) is strongly opposing the weighted sum, reduce confidence
        # Example: ML says SELL (-0.8), but others pull avg to BUY (0.1).
        if ml_sig and ml_sig.confidence > 0.6:
            if (ml_score > 0 and raw_signal < 0) or (ml_score < 0 and raw_signal > 0):
                # Strong disagreement -> Kill signal
                raw_signal = 0.0
                confidence = 0.0
        
        if raw_signal > BUY_THRESH:
            action = TradeAction.BUY
        elif raw_signal < SELL_THRESH:
            action = TradeAction.SELL
            
        # 4. Construct Final Signal
        reasons = []
        reasons.append(f"Grand Score: {raw_signal:.2f}")
        if ml_sig: reasons.append(f"ML: {ml_score:.2f}")
        if deep_sig: reasons.append(f"Deep: {deep_score:.2f}")
        
        intermediate_signal = FinalSignal(
            action=action,
            confidence=round(confidence, 2),
            reasoning=reasons,
            vision_bias=vision.market_bias.value,
            ts_bias=f"Raw:{raw_signal:.2f}",
            regime=regime.state
        )
        
        # 5. Apply Safety Gates (Final Filter)
        final_signal = SafetyGates.validate(intermediate_signal, vision, ts, regime)
        
        return final_signal
