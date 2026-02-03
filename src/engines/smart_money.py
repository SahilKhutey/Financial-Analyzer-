"""
Smart Money / Confluence Engine.
Implements a 6-layer analysis logic:
1. Regime (ATR/Slope)
2. Structure (Support/Resistance)
3. Candle Patterns
4. Momentum (RSI+ROC)
5. Vision (OpenCV Edge Density)
6. Confluence Scoring
"""
import pandas as pd
import numpy as np
import cv2
from typing import Tuple, List, Optional
from src.core.types import FinalSignal, TradeAction

class SmartMoneyEngine:
    """
    Advanced Logic implementing the 'Smart Money' systematic approach.
    """
    
    def __init__(self):
        self.atr_period = 14
        self.slope_period = 20
        self.rsi_period = 14
        
    def _market_regime(self, df: pd.DataFrame) -> str:
        """Layer 1: Detect Market Type"""
        if len(df) < 20: return "UNCERTAIN"
        
        # Calculate ATR
        high_low = df['High'] - df['Low']
        # Approximate rolling ATR for speed
        atr = high_low.rolling(self.atr_period).mean()
        
        # Calculate Slope (Linear Regression slope proxy: (End-Start)/N)
        # Normalized by price to be comparable
        window = df['Close'].rolling(self.slope_period)
        
        # We need the last value
        last_atr = atr.iloc[-1]
        mean_atr = atr.mean()
        
        # Slope of last 20 bars
        start_price = df['Close'].iloc[-20]
        end_price = df['Close'].iloc[-1]
        slope_val = (end_price - start_price) / 20.0
        # Normalize slope by price
        norm_slope = slope_val / end_price
        
        if last_atr > mean_atr * 1.5:
            return "VOLATILE"
        elif abs(norm_slope) > 0.0003: # Threshold from prompt
            return "TREND"
        else:
            return "RANGE"

    def _support_resistance(self, df: pd.DataFrame, window=20) -> Tuple[str, List[float], List[float]]:
        """Layer 2: Structure & Zones"""
        if len(df) < window*2: return "NEUTRAL", [], []
        
        supports = []
        resistances = []
        
        # Scan recent history (optimize loop for just recent history to save time if needed, 
        # but full scan is okay for small frames)
        # Limit to last 100 bars for performance
        subset = df.iloc[-100:] if len(df) > 100 else df
        
        # We need integer indexing for the loop relative to the subset
        closes = subset['Close'].values
        lows = subset['Low'].values
        highs = subset['High'].values
        
        # Simple local extrema
        # Note: The prompt's loop logic checks i-window to i+window. 
        # For the very last bar, we can't check i+window (future).
        # We check verified S/R from the PAST.
        
        for i in range(window, len(subset) - window):
            curr_low = lows[i]
            curr_high = highs[i]
            
            # Check if this is a local min
            if curr_low == min(lows[i-window:i+window]):
                supports.append(curr_low)
                
            # Check if this is a local max
            if curr_high == max(highs[i-window:i+window]):
                resistances.append(curr_high)
        
        # Determine status of Current Price
        current_price = closes[-1]
        
        # Check proximity (within 0.5%)
        bias = "NEUTRAL"
        
        if supports:
            nearest_support = max(supports) # Closest support below? Not necessarily, max of all support levels might be above current if broken.
            # Actually we want the nearest one.
            # Let's just take the last 3 detected
             
        # Simplify PROMPT Logic: "Price near support -> BUY"
        # We look at the most recent detected levels
        recent_supports = supports[-3:] if supports else []
        recent_resistances = resistances[-3:] if resistances else []
        
        is_support = False
        for s in recent_supports:
            if abs(current_price - s) / current_price < 0.005: 
                is_support = True
                
        is_resistance = False
        for r in recent_resistances:
            if abs(current_price - r) / current_price < 0.005:
                is_resistance = True
                
        if is_support: bias = "SUPPORT"
        if is_resistance: bias = "RESISTANCE"
        
        return bias, recent_supports, recent_resistances

    def _candle_pattern(self, row: pd.Series) -> str:
        """Layer 3: Candle Intelligence"""
        body = abs(row['Close'] - row['Open'])
        wick = row['High'] - row['Low']
        
        if wick == 0: return "INDECISION" # Doji-like
        
        if body < wick * 0.3:
            return "INDECISION"
        elif row['Close'] > row['Open']:
            return "BULLISH"
        else:
            return "BEARISH"

    def _momentum(self, df: pd.DataFrame) -> str:
        """Layer 4: Momentum (RSI + ROC)"""
        # Calculate RSI if missing
        if 'rsi' not in df.columns:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
        # ROC (Rate of Change)
        # Prompt: delta / shift(1)
        roc = df['Close'].pct_change()
        
        last_roc = roc.iloc[-1]
        last_rsi = df['rsi'].iloc[-1]
        
        if last_roc > 0 and last_rsi < 65:
            return "BULLISH"
        elif last_roc < 0 and last_rsi > 35:
            return "BEARISH"
        else:
            return "WEAK"

    def _chart_vision(self, image_array: Optional[np.ndarray]) -> str:
        """Layer 5: CV Logic (Edge Density)"""
        if image_array is None: return "NO_IMG"
        
        try:
            # image_array is assumed to be RGB 0-255 or 0-1 from processing
            # Check type. If float 0-1 (from Neural Net prep), scale up.
            img = image_array
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
                
            # Convert to Gray if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif len(img.shape) == 3 and img.shape[2] == 1:
                gray = img[:,:,0]
            else:
                gray = img
            
            # Canny
            edges = cv2.Canny(gray, 50, 150)
            density = np.sum(edges) / edges.size
            
            # Thresholds from prompt
            if density > 0.08:
                return "STRONG_MOVE"
            elif density < 0.03:
                return "RANGE"
            else:
                return "NORMAL"
        except Exception as e:
            print(f"Vision Error: {e}")
            return "ERROR"

    def analyze(self, df: pd.DataFrame, vision_image: Optional[np.ndarray] = None) -> FinalSignal:
        """Layer 6: Confluence Engine"""
        if df is None or len(df) < 50:
            return FinalSignal(TradeAction.STAY_OUT, 0.0, ["Insufficient Data"], "N/A", "N/A", "N/A")
            
        # 1. Regime
        regime = self._market_regime(df)
        if regime == "VOLATILE":
             return FinalSignal(TradeAction.STAY_OUT, 0.0, ["Market Volatile - HALT"], "N/A", "N/A", regime)
             
        # 2. Structure
        struct_bias, _, _ = self._support_resistance(df)
        
        # 3. Candle
        candle = self._candle_pattern(df.iloc[-1])
        
        # 4. Momentum
        moment = self._momentum(df)
        
        # 5. Vision
        # Image passed might be (1, 224, 224, 3) from NN wrapper or raw. 
        # If it's 4D tensor, take first item.
        vis_res = "NO_IMG"
        if vision_image is not None:
            if vision_image.ndim == 4:
                vis_img = vision_image[0]
            else:
                vis_img = vision_image
            vis_res = self._chart_vision(vis_img)
            
        # 6. Scoring
        score = 0
        reasons = []
        
        if regime == "TREND":
            score += 2
            reasons.append("Trend Regime (+2)")
        
        if struct_bias == "SUPPORT":
            score += 2
            reasons.append("At Support (+2)")
        elif struct_bias == "RESISTANCE":
             score -= 2 # Sell bias? 
             # Prompt says "Price near resistance -> SELL bias"
             # Let's assume negative score is SELL
             reasons.append("At Resistance (-2)")
             
        if candle == "BULLISH":
            score += 1
            reasons.append("Bullish Candle (+1)")
        elif candle == "BEARISH":
            score -= 1
            reasons.append("Bearish Candle (-1)")
            
        if moment == "BULLISH":
            score += 1
            reasons.append("Bull Momentum (+1)")
        elif moment == "BEARISH":
            score -= 1
            reasons.append("Bear Momentum (-1)")
            
        if vis_res == "STRONG_MOVE":
            # Ambiguous: Strong move UP or DOWN? 
            # Canny just sees edges (activity).
            # Usually implies momentum continuation. 
            # We'll align it with the current score direction.
            if score > 0: 
                score += 1
                reasons.append("Vision: Strong Activity (+1)")
            elif score < 0:
                score -= 1
                reasons.append("Vision: Strong Activity (-1)")
        
        # Final Decision
        action = TradeAction.STAY_OUT
        confidence = 0.0
        
        if score >= 4: # Prompt says >= 5, but let's be slightly more lenient for testing or stick to 5
             # Prompt: >= 5 -> BUY
             if score >= 5:
                 action = TradeAction.BUY
                 confidence = min(0.99, score * 0.1) # Score 5 -> 50%, Score 8 -> 80%
             else:
                 # Weak Buy
                 action = TradeAction.STAY_OUT # Strict rule: "Confirm with at least 3 confluences" -> score 5 implies specific mix
                 confidence = score * 0.1
                 
        elif score <= -5:
            action = TradeAction.SELL
            confidence = min(0.99, abs(score) * 0.1)
            
        else:
            action = TradeAction.STAY_OUT
            confidence = abs(score) * 0.1
            
        return FinalSignal(
            action=action,
            confidence=round(confidence, 2),
            reasoning=reasons,
            vision_bias=vis_res,
            ts_bias=moment,
            regime=regime
        )
