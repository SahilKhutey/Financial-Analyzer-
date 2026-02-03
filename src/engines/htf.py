"""
Step 9: Multi-Timeframe Analysis (HTF).
Focus: Weekly/Daily Trend Alignment (Fractal Analysis).
"""

import pandas as pd
from src.core.types import MarketBias

class MultiTimeframeEngine:
    """
    Analyzes Higher Timeframe (Weekly) context.
    """
    
    def analyze_weekly(self, df_weekly: pd.DataFrame) -> MarketBias:
        """
        Determine the heavy macro bias.
        """
        if len(df_weekly) < 20:
            return MarketBias.NEUTRAL
            
        # Simple Logic: Price relative to SMA 20 (Weekly) ~ approx SMA 100 Daily
        closes = df_weekly['Close']
        sma_20 = closes.rolling(20).mean().iloc[-1]
        last_close = closes.iloc[-1]
        
        # Trend check
        if last_close > sma_20 * 1.01:
            return MarketBias.BULLISH
        elif last_close < sma_20 * 0.99:
            return MarketBias.BEARISH
        else:
            return MarketBias.NEUTRAL
