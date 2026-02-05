"""
Step 4: Market Regime Filter.
"The Gatekeeper" - Determines if the market environment is safe for trading.
Rejects trades during High Volatility, Low Liquidity, or Chaotic conditions.
"""

import pandas as pd
import numpy as np
from src.core.types import RegimeOutput
from src.core.config import MAX_DAILY_DRAWDOWN_LIMIT

from sklearn.mixture import GaussianMixture

class RegimeFilter:
    """
    Analyzes volatility and volume profiles to classify market safety.
    Upgraded: Uses Gaussian Mixture Models (GMM) for probabilistic regime detection.
    """
    
    def __init__(self):
        # 3 States: Range (Low Vol), Trend (Med Vol), Volatile (High Vol)
        self.gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
        self.is_fitted = False
    
    def analyze(self, df: pd.DataFrame, macro_state=None) -> RegimeOutput:
        """
        Check if market is safe.
        """
        if len(df) < 30:
            return RegimeOutput(False, "Insufficient Data", "Need >30 bars")
            
        last = df.iloc[-1]
        
        # 1. Macro Safety Check (Global Veto)
        if macro_state and macro_state.regime == "RECESSION":
             return RegimeOutput(False, "MACRO: RECESSION", "Global Risk Veto")
        
        # 2. Hard Gate: Extreme Volatility Check (ATR)
        if 'atr' not in df.columns:
            tr = pd.concat([
                df['High'] - df['Low'], 
                (df['High'] - df['Close'].shift()).abs(), 
                (df['Low'] - df['Close'].shift()).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
        else:
            atr = df['atr'].iloc[-1]
            
        rel_atr = atr / last['Close']
        if rel_atr > MAX_DAILY_DRAWDOWN_LIMIT:
             return RegimeOutput(False, "High Volatility", f"ATR {rel_atr:.1%} > Limit")
             
        # 3. Hard Gate: Liquidity Check (Volume)
        avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
        if last['Volume'] < (avg_vol * 0.1): # 10% threshold (institutional)
             return RegimeOutput(False, "Dead Market", "Volume < 10% Avg")
             
        # 4. GMM Regime Classification
        # We classify based on Returns distribution
        returns = df['Close'].pct_change().dropna().values.reshape(-1, 1)
        state_label = "NORMAL"
        
        try:
            if len(returns) > 50:
                self.gmm.fit(returns)
                curr_state = self.gmm.predict(returns[-1].reshape(1, -1))[0]
                
                # Identify regimes by Variance (Volatility)
                vars = self.gmm.covariances_.flatten()
                sorted_vars = np.sort(vars)
                
                curr_var = vars[curr_state]
                
                if curr_var == sorted_vars[0]:
                    state_label = "RANGING" # Lowest Vol
                elif curr_var == sorted_vars[2]:
                    state_label = "VOLATILE" # Highest Vol
                else:
                    state_label = "TRENDING" # Medium Vol
        except Exception:
            state_label = "UNCERTAIN" # Fallback
            
        # Append Macro tag if Risk-Off
        if macro_state and macro_state.regime == "RISK_OFF":
            state_label += " ( RISK-OFF )"
            
        return RegimeOutput(True, state_label, "Vol & Liq Safe")
