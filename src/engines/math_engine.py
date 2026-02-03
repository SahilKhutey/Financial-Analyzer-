"""
Mathematical Market Prediction Engine.
(Short-term / Binary / 1m–5m charts)

Core Truth: Markets are stochastic processes.
We predict PROBABILITY of direction within Delta_t.

Layers:
1. Stochastic Process (Drift & Volatility)
2. Directional Probability (Normal Approximation)
3. Entropy (Market Clarity)
4. Hurst Exponent (Trend vs Mean Reversion)
5. Momentum Physics (Velocity & Acceleration)
6. Bayesian Update (Prior/Posterior Belief)
7. Markov Transition Matrix (State Memory)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Tuple, List, Optional
from src.core.types import FinalSignal, TradeAction

class MathPredictionEngine:
    """
    Advanced Mathematical Prediction Engine for short-term probabilistic forecasting.
    """
    
    def __init__(self):
        self.lookback = 20
        self.hurst_min_lags = 2
        self.hurst_max_lags = 20

    # 1. Stochastic Process Parameters
    def _stochastic_params(self, returns: pd.Series) -> Tuple[float, float]:
        """
        Model price as Geometric Random Walk.
        Calculate Drift (mu) and Volatility (sigma).
        """
        # Mu = mean of log returns
        mu = returns.rolling(self.lookback).mean().iloc[-1]
        # Sigma = std of log returns
        sigma = returns.rolling(self.lookback).std().iloc[-1]
        return mu, sigma

    # 2. Directional Probability
    def _directional_probability(self, mu: float, sigma: float) -> float:
        """
        P(r > 0) using Normal Approximation.
        """
        if pd.isna(mu) or pd.isna(sigma) or sigma == 0:
            return 0.5
        # P = 1 - CDF(0, mu, sigma)
        prob = 1 - norm.cdf(0, mu, sigma)
        return prob

    # 3. Entropy
    def _entropy(self, returns: pd.Series) -> float:
        """
        Shannon Entropy to measure market disorder/unpredictability.
        """
        subset = returns.iloc[-self.lookback:]
        p_up = np.mean(subset > 0)
        p_down = 1 - p_up
        
        if p_up <= 0 or p_down <= 0:
            return 0.0 # Zero entropy (fully deterministic in this window)
            
        entropy = -(p_up * np.log(p_up) + p_down * np.log(p_down))
        return entropy

    # 4. Hurst Exponent
    def _hurst_exponent(self, prices: pd.Series) -> float:
        """
        Measure long-term memory of time series.
        H > 0.5: Trending
        H < 0.5: Mean Reversion
        """
        ts = prices.values
        if len(ts) < self.hurst_max_lags * 2:
            return 0.5
            
        lags = range(self.hurst_min_lags, self.hurst_max_lags)
        tau = []
        for lag in lags:
            # Standard deviation of differences
            diff = ts[lag:] - ts[:-lag]
            tau.append(np.std(diff))
        
        # Avoid log(0)
        if any(t == 0 for t in tau):
            return 0.5
            
        # Polyfit log(lags) vs log(tau)
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]

    # 5. Momentum Physics
    def _momentum_physics(self, prices: pd.Series) -> Tuple[float, float]:
        """
        Velocity (1st derivative) and Acceleration (2nd derivative).
        """
        velocity = prices.diff()
        acceleration = velocity.diff()
        
        v = velocity.iloc[-1]
        a = acceleration.iloc[-1]
        
        return v, a

    # 6. Bayesian Update
    def _bayesian_update(self, prior: float, likelihood: float) -> float:
        """
        Update belief P(BUY | Evidence).
        """
        numerator = likelihood * prior
        denominator = (likelihood * prior) + ((1 - likelihood) * (1 - prior))
        
        if denominator == 0:
            return prior
            
        return numerator / denominator

    # 7. Markov Chain
    def _markov_matrix(self, returns: pd.Series) -> Tuple[float, float, float]:
        """
        Calculate Transition Probabilities for 3-State Chain: Down, Flat, Up.
        Returns: P(Down|Current), P(Flat|Current), P(Up|Current)
        """
        subset = returns.iloc[-100:] # Need more history for 3 states
        threshold = subset.abs().mean() * 0.5 # Dynamic threshold for "Flat"
        
        # Digitize states: 0=Down, 1=Flat, 2=Up
        states = []
        vals = subset.values
        for r in vals:
            if r > threshold: states.append(2)
            elif r < -threshold: states.append(0)
            else: states.append(1)
        
        states = np.array(states)
        current_state = states[-1]
        
        # Calculate transitions from Current State
        # Find all indices where we were in the current state (excluding last point)
        indices = np.where(states[:-1] == current_state)[0]
        
        if len(indices) < 3:
            return 0.33, 0.33, 0.33 # Insufficient data
            
        next_states = states[indices + 1]
        
        # Count next states
        n_down = np.sum(next_states == 0)
        n_flat = np.sum(next_states == 1)
        n_up = np.sum(next_states == 2)
        total = len(next_states)
        
        return n_down/total, n_flat/total, n_up/total

    def analyze(self, df: pd.DataFrame, vision_out=None) -> FinalSignal:
        """
        Main Analysis Pipeline.
        """
        if df is None or len(df) < 50:
            return FinalSignal(TradeAction.STAY_OUT, 0.0, ["Insufficient Data"], "N/A", "N/A", "N/A")
            
        # Prepare Data
        closes = df['Close']
        returns = np.log(closes / closes.shift(1))
        
        # 1. Stochastic Params
        mu, sigma = self._stochastic_params(returns)
        
        # 2. Directional Prob (Likelihood)
        prob_raw = self._directional_probability(mu, sigma)
        
        # 6. Bayesian Update (Prior = 0.5 Neutral)
        # We treat the raw Normal prob as the likelihood of UP move
        bayesian_prob = self._bayesian_update(0.5, prob_raw)
        
        # 3. Entropy
        entropy_val = self._entropy(returns)
        
        # 4. Hurst
        hurst_val = self._hurst_exponent(closes)
        
        # 5. Physics
        vel, accel = self._momentum_physics(closes)
        
        # 7. Markov
        prob_d, prob_f, prob_u = self._markov_matrix(returns)
        
        # Scoring Logic
        score = 0
        reasons = []
        
        # A. Probability Score
        if bayesian_prob > 0.6:
            score += 2
            reasons.append(f"High Prob (Bayes: {bayesian_prob:.2f})")
        elif bayesian_prob < 0.4:
            score -= 2
            reasons.append(f"Low Prob (Bayes: {bayesian_prob:.2f})")
            
        # B. Entropy Filter
        # Low entropy = predictable
        if entropy_val < 0.6: # Config threshold
            score += 1 if score >= 0 else -1 # Boost existing bias
            reasons.append(f"Low Entropy ({entropy_val:.2f})")
        else:
            # High entropy penalizes confidence or just doesn't add score
            reasons.append(f"High Entropy ({entropy_val:.2f})")
            
        # C. Hurst (Trend Confirmation)
        if hurst_val > 0.55:
            reasons.append(f"Trending (H: {hurst_val:.2f})")
            # If following probability direction
            if bayesian_prob > 0.5 and score > 0: score += 1
            if bayesian_prob < 0.5 and score < 0: score -= 1
        elif hurst_val < 0.45:
             reasons.append(f"Mean Rev (H: {hurst_val:.2f})")
             
        # D. Physics (Acceleration)
        if vel > 0 and accel > 0:
            score += 1
            reasons.append("Accel: UP")
        elif vel < 0 and accel < 0:
            score -= 1
            reasons.append("Accel: DOWN")
            
        # E. Markov Prediction (3-State)
        if prob_u > 0.5:
             score += 1
             reasons.append(f"Markov: UP ({prob_u:.2f})")
        elif prob_d > 0.5:
             score -= 1
             reasons.append(f"Markov: DOWN ({prob_d:.2f})")
        elif prob_f > 0.6:
             reasons.append(f"Markov: FLAT ({prob_f:.2f})")
             # Penalty for flat predictions if trying to trade?
             # For now, it just doesn't add directional score.
        
        # Final Decision
        action = TradeAction.STAY_OUT
        confidence = 0.0
        
        # Thresholds
        if score >= 4:
            action = TradeAction.BUY
            confidence = min(0.99, score * 0.15)
        elif score <= -4:
            action = TradeAction.SELL
            confidence = min(0.99, abs(score) * 0.15)
        else:
            confidence = abs(score) * 0.1
            
        return FinalSignal(
            action=action,
            confidence=round(confidence, 2),
            reasoning=reasons[:3], # Top 3 reasons
            vision_bias="N/A",
            ts_bias=f"Prob:{bayesian_prob:.2f}",
            regime=f"H={hurst_val:.2f}"
        )
