"""
Step 32: Global Macro Engine.
Simulates global economic factors (DXY, VIX, Yields) to determine Risk-On/Risk-Off bias.
"""
from dataclasses import dataclass
import random

@dataclass
class MacroState:
    dxy: float        # Dollar Index (e.g., 104.5)
    vix: float        # Volatility Index (e.g., 15.2)
    us10y: float      # 10-Year Treasury Yield (e.g., 4.2%)
    regime: str       # "RISK_ON", "RISK_OFF", "NEUTRAL", "RECESSION"
    multiplier: float # Position sizing multiplier (e.g., 1.2x, 0.5x)

class MacroEngine:
    """
    Monitors global financial conditions.
    """
    
    def __init__(self):
        # Simulated base values
        self.base_dxy = 104.0
        self.base_vix = 18.0
        self.base_us10y = 4.2
        
    def analyze(self) -> MacroState:
        """
        Get current global macro state.
        In production, this would scrape MarketWatch/YahooFinance.
        """
        # Simulate slight random fluctuation
        dxy = self.base_dxy + random.uniform(-0.5, 0.5)
        vix = self.base_vix + random.uniform(-2.0, 5.0) # VIX spikes easier
        us10y = self.base_us10y + random.uniform(-0.1, 0.1)
        
        # Determine Regime
        # Logic: 
        # Risk-Off if DXY Strong (>105) or VIX High (>25)
        # Risk-On if DXY Weak (<103) and VIX Low (<20)
        
        regime = "NEUTRAL"
        multiplier = 1.0
        
        if vix > 35.0:
            regime = "RECESSION"
            multiplier = 0.0 # Cash is King
        elif vix > 25.0 or dxy > 105.0:
            regime = "RISK_OFF"
            multiplier = 0.5 # Defensive half-size
        elif vix < 20.0 and dxy < 103.5:
            regime = "RISK_ON"
            multiplier = 1.25 # Aggressive
        else:
            regime = "NEUTRAL"
            multiplier = 1.0
            
        return MacroState(
            dxy=round(dxy, 2),
            vix=round(vix, 2),
            us10y=round(us10y, 3),
            regime=regime,
            multiplier=multiplier
        )
