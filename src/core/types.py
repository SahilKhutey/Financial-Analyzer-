"""
Core Data Types.
Defines the standard data structures exchanged between engines.
"""
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

class MarketBias(Enum):
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"

class TradeAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    STAY_OUT = "STAY_OUT"

@dataclass
class VisionOutput:
    """Output from the Vision Engine."""
    market_bias: MarketBias
    breakout_prob: float      # 0.0 - 1.0
    reversal_prob: float      # 0.0 - 1.0
    momentum_score: float     # 0.0 (Low) - 1.0 (High)
    patterns_detected: List[str]

@dataclass
class TSOutput:
    """Output from the Time-Series Engine."""
    bullish_prob: float       # 0.0 - 1.0
    bearish_prob: float       # 0.0 - 1.0
    forecast_volatility: float

@dataclass
class RegimeOutput:
    """Output from the Regime Filter."""
    is_safe: bool
    state: str                # e.g., "Trending", "High_Vol"
    reason: str

@dataclass
class FinalSignal:
    """Final fused signal for the user."""
    action: TradeAction
    confidence: float
    reasoning: List[str]
    vision_bias: str
    ts_bias: str
    regime: str
    model_votes: Optional[dict] = None
