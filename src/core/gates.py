"""
Step 6: Safety & Logic Gates.
Non-negotiable final checks before a signal is approved.
"""

from src.core.types import FinalSignal, VisionOutput, TSOutput, RegimeOutput, TradeAction
from src.core.config import CONFIDENCE_THRESHOLD

class SafetyGates:
    """
    Applies institutional logic rules to filter signals.
    """
    
    @staticmethod
    def validate(
        signal: FinalSignal, 
        vision: VisionOutput, 
        ts: TSOutput, 
        regime: RegimeOutput
    ) -> FinalSignal:
        """
        Run the Gauntlet.
        """
        # Rule 1: Regime Check (Institutional)
        if not regime.is_safe:
             return FinalSignal(TradeAction.STAY_OUT, 0.0, [f"Regime Lock: {regime.state} ({regime.reason})"], signal.vision_bias, signal.ts_bias, regime.state)
        
        # New Rule: Deep Analytic Regime Check
        # If the comprehensive regime filter detects Volatile, kill it.
        if "VOLATILE" in str(signal.reasoning):
             return FinalSignal(TradeAction.STAY_OUT, 0.0, ["Deep Regime: VOLATILE"], signal.vision_bias, signal.ts_bias, regime.state)

        # Rule 2: Minimum Confidence (Dynamic)
        threshold = CONFIDENCE_THRESHOLD
        
        # Stricter for basic signals, ML signals carry their own high confidence
        if signal.confidence < threshold:
            return FinalSignal(
                action=TradeAction.STAY_OUT,
                confidence=signal.confidence,
                reasoning=[f"Confidence {signal.confidence:.2f} < Threshold {threshold}"],
                vision_bias=signal.vision_bias,
                ts_bias=signal.ts_bias,
                regime=regime.state
            )
            
        # Rule 3: Conflict Check (Vision vs TS)
        # If Vision says Bull and TS says Bear (Strong prob > 0.6), conflict!
        vision_bull = vision.market_bias == "Bullish" 
        ts_bear = ts.bearish_prob > 0.6
        
        if vision_bull and ts_bear and signal.confidence < 0.8: # Allow ML override if super confident mismatch
             return FinalSignal(
                action=TradeAction.STAY_OUT,
                confidence=0.0,
                reasoning=["Conflict: Vision Bullish vs TS Bearish"],
                vision_bias=signal.vision_bias,
                ts_bias=signal.ts_bias,
                regime=regime.state
            )
            
        # Rule 4: ML Validation
        # If this is an ML signal, ensure we trust the probability
        if "XGB" in str(signal.reasoning) and signal.confidence < 0.65:
             # Basic Threshold is 0.6, but for ML we want 0.65 for entry
             return FinalSignal(TradeAction.STAY_OUT, signal.confidence, ["ML Conf < 0.65"], signal.vision_bias, signal.ts_bias, regime.state)

        # If passed all:
        return signal
