"""
Complete Deep-Analytical Signal Engine.
(Binary / Binomo / 1m-5m Institutional Modules)

Layers:
1. Kalman Filter (True Trend Extraction)
2. HMM/GMM (Market Regime Detection) - Using GaussianMixture as robust fallback for HMM
3. Directional Probability (Stochastic Drift)
4. Entropy (Noise Filter)
5. Bayesian Fusion (Final Probabilistic Signal)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Tuple, List, Optional
from src.core.types import FinalSignal, TradeAction

# Attempt imports with safe fallbacks
try:
    from pykalman import KalmanFilter
    HAS_KALMAN = True
except ImportError:
    HAS_KALMAN = False
    print("Warning: pykalman not found. Using simple fallback.")

from sklearn.mixture import GaussianMixture

class DeepAnalyticalEngine:
    """
    Institutional-Grade Signal Engine.
    Fuses 5 advanced analytical modules.
    """
    
    def __init__(self):
        self.lookback = 30
        self.kf = None
        if HAS_KALMAN:
            # Initialize Kalman Filter
            # Transition=1 (Random Walk), Observation=1
            self.kf = KalmanFilter(
                transition_matrices=[1],
                observation_matrices=[1],
                initial_state_mean=0,
                initial_state_covariance=1,
                observation_covariance=1,
                transition_covariance=0.01
            )
        
        # GMM for Regime (3 states: Trend, Range, Volatile)
        self.gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)

    # 1. Kalman Filter (Trend)
    def _kalman_trend(self, prices: pd.Series) -> int:
        """
        Extract True Trend using Kalman Smoother.
        Returns: 1 (Up), -1 (Down), 0 (Neutral)
        """
        if not HAS_KALMAN:
            # Fallback: Simple EMA
            return 1 if prices.iloc[-1] > prices.ewm(span=20).mean().iloc[-1] else -1
            
        try:
            # Adaptive Logic: Tune covariance based on recent volatility
            # High Vol -> Higher Covariance (Follow price closely)
            # Low Vol -> Lower Covariance (Smooth out noise)
            returns = prices.pct_change().dropna()
            vol = returns.std()
            
            # Map vol (0.0001 - 0.01) to cov (0.001 - 0.1)
            # Scaling factor heuristic
            trans_cov = max(0.001, min(0.1, vol * 10))
            
            kf = KalmanFilter(
                transition_matrices=[1],
                observation_matrices=[1],
                initial_state_mean=prices.values[0],
                initial_state_covariance=1,
                observation_covariance=1,
                transition_covariance=trans_cov
            )
            
            state_means, _ = kf.filter(prices.values)
            # Calculate gradient of the hidden state (true price)
            trend = np.gradient(state_means.flatten())
            
            last_trend = trend[-1]
            if last_trend > 0.05: return 1
            elif last_trend < -0.05: return -1
            else: return 0
        except Exception as e:
            print(f"Kalman Error: {e}")
            return 0

    # 2. Regime Detection (GMM)
    def _detect_regime(self, returns: pd.Series) -> str:
        """
        Cluster market state into 3 regimes.
        We assume:
        - Low variance = Range
        - Medium variance, high directional mean = Trend
        - High variance = Volatile
        """
        # Reshape for sklearn
        X = returns.values.reshape(-1, 1)
        if len(X) < 50: return "UNCERTAIN"
        
        try:
            self.gmm.fit(X)
            hidden_states = self.gmm.predict(X)
            current_state = hidden_states[-1]
            
            # Analyze state properties to label them
            means = self.gmm.means_.flatten()
            vars = self.gmm.covariances_.flatten()
            
            # Identify current state params
            curr_var = vars[current_state]
            
            # Simple heuristic mapping based on relative variance
            sorted_vars = np.sort(vars)
            
            if curr_var == sorted_vars[0]: # Lowest variance
                return "RANGE"
            elif curr_var == sorted_vars[2]: # Highest variance
                return "VOLATILE"
            else:
                return "TREND"
        except Exception:
            return "UNCERTAIN"

    # 3. Directional Probability
    def _direction_prob(self, returns: pd.Series) -> float:
        """
        P(Return > 0) using Normal Approximation.
        """
        mu = returns.mean()
        sigma = returns.std()
        if sigma == 0: return 0.5
        return 1 - norm.cdf(0, mu, sigma)

    # 4. Entropy Filter
    def _entropy(self, returns: pd.Series) -> float:
        """
        Shannon Entropy (Noise level).
        """
        p_up = np.mean(returns > 0)
        p_down = 1 - p_up
        if p_up <= 0 or p_down <= 0: return 0.0
        return -(p_up * np.log(p_up) + p_down * np.log(p_down))

    # 5. Bayesian Fusion
    def _bayesian_fusion(self, probs: List[float], weights: List[float]) -> float:
        """
        Combine probabilities using Log-Odds.
        """
        log_odds = 0
        for p, w in zip(probs, weights):
            # Clip p to avoid log(0)
            p = max(0.01, min(0.99, p))
            log_odds += w * np.log(p / (1 - p))
            
        return 1 / (1 + np.exp(-log_odds))

    def analyze(self, df: pd.DataFrame, vision_out=None) -> FinalSignal:
        """
        Main Pipeline.
        """
        if df is None or len(df) < 50:
            return FinalSignal(TradeAction.STAY_OUT, 0.0, ["Insufficient Data"], "N/A", "N/A", "N/A")
            
        # Prep
        prices = df['Close']
        returns = np.log(prices / prices.shift(1)).dropna()
        
        # 1. Kalman Trend
        k_trend = self._kalman_trend(prices) # 1, -1, 0
        
        # 2. Regime
        regime = self._detect_regime(returns)
        if regime == "VOLATILE":
             return FinalSignal(TradeAction.STAY_OUT, 0.0, [f"Regime: {regime} (Unsafe)"], "N/A", "N/A", regime)
             
        # 3. Probability
        prob_dir = self._direction_prob(returns.tail(30))
        
        # 4. Entropy
        entropy = self._entropy(returns.tail(50))
        is_safe_entropy = entropy < 0.68 # Slightly loose threshold
        
        if not is_safe_entropy:
             return FinalSignal(TradeAction.STAY_OUT, 0.0, [f"High Entropy: {entropy:.2f}"], "N/A", f"Prob:{prob_dir:.2f}", regime)
             
        # 5. Bayesian Fusion
        # Inputs: 
        # A: Direction Probability
        # B: Kalman Bias (converted to prob: 1->0.6, -1->0.4, 0->0.5)
        
        k_prob = 0.5
        if k_trend == 1: k_prob = 0.65
        elif k_trend == -1: k_prob = 0.35
        
        fused_prob = self._bayesian_fusion(
            probs=[prob_dir, k_prob],
            weights=[0.6, 0.4] # Prob has slightly more weight than Trend bias
        )
        
        # Final Decision
        action = TradeAction.STAY_OUT
        confidence = 0.0
        reasons = []
        
        reasons.append(f"Fused Prob: {fused_prob:.2f}")
        reasons.append(f"Regime: {regime}")
        reasons.append(f"Entropy: {entropy:.2f}")
        
        if fused_prob > 0.60:
            action = TradeAction.BUY
            confidence = fused_prob
        elif fused_prob < 0.40:
            action = TradeAction.SELL
            confidence = 1 - fused_prob
        else:
            action = TradeAction.STAY_OUT
            confidence = max(fused_prob, 1-fused_prob)
            
        return FinalSignal(
            action=action,
            confidence=round(confidence, 2),
            reasoning=reasons,
            vision_bias="N/A",
            ts_bias=f"Kalman:{k_trend}",
            regime=regime
        )
