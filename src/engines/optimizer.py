"""
Step 26: Intelligent Optimization Engine.
Performs simulation runs (Grid Search) to find optimal Fusion Weights.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from src.engines.backtest import BacktestEngine
from src.core.config import FUSION_WEIGHTS

class OptimizerEngine:
    """
    Self-Adaptive Tuning Module.
    Runs simulations on recent data to find optimal weight configurations.
    """
    
    def __init__(self):
        self.backtester = BacktestEngine()
        
    def generate_grid(self) -> List[Dict[str, float]]:
        """
        Generate experimental weight configurations.
        Includes commonly effective 'Personas'.
        """
        grid = []
        
        # 1. Default (Balanced)
        grid.append(FUSION_WEIGHTS.copy())
        
        # 2. Vision Heavy (for visual trends)
        grid.append({
            'ml': 0.20, 'deep': 0.15, 'math': 0.10, 'vision': 0.45, 'ts': 0.10
        })
        
        # 3. ML Heavy (for algorithmic markets)
        grid.append({
            'ml': 0.50, 'deep': 0.20, 'math': 0.10, 'vision': 0.10, 'ts': 0.10
        })
        
        # 4. Institutional (Deep Learning Focus)
        grid.append({
            'ml': 0.20, 'deep': 0.50, 'math': 0.10, 'vision': 0.10, 'ts': 0.10
        })
        
        # 5. Math Pure (Oscillators)
        grid.append({
            'ml': 0.10, 'deep': 0.10, 'math': 0.50, 'vision': 0.10, 'ts': 0.20
        })
        
        return grid
        
    def run_optimization(self, df: pd.DataFrame, top_n: int = 1) -> List[Dict]:
        """
        Run backtests for each grid configuration.
        Return ranked results.
        """
        if len(df) < 200:
            return []
            
        print(f"🧠 Optimizer: Starting Grid Search on {len(df)} candles...")
        
        results = []
        configs = self.generate_grid()
        
        # Use a slice for speed (last 500 candles usually sufficient for Regime extraction)
        # Note: Backtester requires warmup, so we pass enough data
        test_df = df.iloc[-500:] if len(df) > 500 else df
        
        for i, weights in enumerate(configs):
            # 1. Inject Weights into Fusion Engine temporarily
            # We access the fusion engine instance inside backtester
            self.backtester.fusion.set_weights(weights)
            
            # 2. Run Sim
            # "Auto (Fusion)" mode will use the weights we just set
            trades = self.backtester.run(test_df, strategy_mode="Auto (Fusion)")
            
            # 3. Score
            metrics = self.backtester.calculate_metrics(trades, 10000.0)
            
            # Parse 'Total Return' string to float for sorting
            ret_str = metrics['Total Return'].replace('%', '')
            score = float(ret_str)
            
            # Name the config
            name = "Custom"
            if i == 0: name = "Default (Balanced)"
            elif weights['vision'] > 0.4: name = "Vision Heavy"
            elif weights['ml'] > 0.4: name = "ML Heavy"
            elif weights['deep'] > 0.4: name = "Deep Institutional"
            elif weights['math'] > 0.4: name = "Math Focus"
            
            results.append({
                "name": name,
                "score": score,
                "sharpe": float(metrics['Sharpe']),
                "win_rate": metrics['Win Rate'],
                "weights": weights,
                "metrics": metrics
            })
            
        # Sort by Score (Return)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:top_n]
