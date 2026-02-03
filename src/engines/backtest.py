"""
Step 8: Enterprise Backtest Engine.
Simulates Institutional Strategies over historical data.
"""

import pandas as pd
import numpy as np
from src.core.types import FinalSignal, TradeAction, VisionOutput, TSOutput, RegimeOutput, MarketBias
from src.engines.vision import EfficientNetWrapper
from src.engines.timeseries import LSTMModel
from src.engines.regime import RegimeFilter
from src.engines.fusion import SignalFusion

# Institutional Engines
from src.engines.math_engine import MathPredictionEngine
from src.engines.deep_analytic import DeepAnalyticalEngine
from src.engines.ml_production import MLProductionEngine

# Specialized
from src.engines.smart_money import SmartMoneyEngine

class BacktestEngine:
    """
    Enterprise-Grade Backtesting Suite.
    Replays market history through the entire pipeline, including Online Learning simulation.
    """
    
    def __init__(self):
        # 1. Base Engines
        self.vision = EfficientNetWrapper()
        self.ts = LSTMModel()
        self.regime = RegimeFilter()
        self.fusion = SignalFusion()
        
        # 2. Institutional Engines
        self.math_engine = MathPredictionEngine()
        self.deep_engine = DeepAnalyticalEngine()
        self.ml_engine = MLProductionEngine()
        self.smart_engine = SmartMoneyEngine()
        
    def run(self, df: pd.DataFrame, initial_capital: float = 10000.0, strategy_mode: str = "Auto (Fusion)") -> pd.DataFrame:
        """
        Execute backtest with specified strategy.
        Results include detailed trade log.
        """
        if len(df) < 200:
            return pd.DataFrame()
            
        results = []
        equity = initial_capital
        position = 0 # 0=Flat, 1=Long, -1=Short
        entry_price = 0.0
        
        # Reset ML Engine for fresh simulation
        self.ml_engine = MLProductionEngine() 
        
        print(f"🎬 Starting Backtest: {strategy_mode} on {len(df)} candles...")
        
        # Walk Forward Loop
        # We start at index 200 to allow warm-up for ML/Indicators
        for i in range(200, len(df)):
            # Window for analysis (simulate "live" data arriving)
            # ML needs full history up to now to train/predict
            window = df.iloc[:i+1] # Pass full growing window, engine handles lookback
            current_bar = df.iloc[i]
            idx = df.index[i]
            
            # --- 1. Generate Signal ---
            signal = None
            
            # Legacy Outputs (always computed for base fusion)
            vision_out = self._mock_vision_history(window.iloc[-50:])
            ts_out = self.ts.predict(window.iloc[-50:])
            regime_out = self.regime.analyze(window.iloc[-50:])
            
            if strategy_mode == "Auto (Fusion)":
                # Compute Institutional Inputs
                ml_sig = self.ml_engine.analyze(window)
                deep_sig = self.deep_engine.analyze(window)
                math_sig = self.math_engine.analyze(window)
                
                signal = self.fusion.fuse(
                    vision_out, ts_out, regime_out,
                    ml_sig=ml_sig, deep_sig=deep_sig, math_sig=math_sig
                )
                
            elif strategy_mode == "Enterprise ML (XGB+LSTM)":
                signal = self.ml_engine.analyze(window)
                
            elif strategy_mode == "Deep Analytical (Institutional)":
                signal = self.deep_engine.analyze(window)
                
            elif strategy_mode == "Smart Money":
                signal = self.smart_engine.analyze(window) # Assumes it handles window slicing
                
            else: # Default Legacy
                signal = self.fusion.fuse(vision_out, ts_out, regime_out)

            # --- 2. Execution Logic (Vectorized would be faster, but this is event-driven) ---
            price = current_bar['Close']
            
            # Check Stop Loss / Take Profit (Simplified)
            # In a real engine, we'd check High/Low of the candle
            
            # Exit Conditions
            if position != 0:
                is_reversal = False
                if position == 1 and signal.action == TradeAction.SELL: is_reversal = True
                elif position == -1 and signal.action == TradeAction.BUY: is_reversal = True
                
                # Force exit on STAY_OUT if specifically enabled, otherwise hold trend
                # Institutional style: often hold until opposite signal or stop
                
                if is_reversal or signal.action == TradeAction.STAY_OUT:
                    # Close Trade
                    if position == 1:
                        pnl_pct = (price - entry_price) / entry_price
                        pnl_amt = pnl_pct * equity
                        equity += pnl_amt
                        results.append({'Date': idx, 'Type': 'EXIT_LONG', 'Price': price, 'Eq': equity, 'PnL': pnl_pct})
                        
                    elif position == -1:
                        pnl_pct = (entry_price - price) / entry_price
                        pnl_amt = pnl_pct * equity
                        equity += pnl_amt
                        results.append({'Date': idx, 'Type': 'EXIT_SHORT', 'Price': price, 'Eq': equity, 'PnL': pnl_pct})
                    
                    position = 0
            
            # Entry Conditions
            if position == 0:
                if signal.confidence > 0.6: # Configurable threshold
                    if signal.action == TradeAction.BUY:
                        position = 1
                        entry_price = price
                        results.append({'Date': idx, 'Type': 'ENTRY_LONG', 'Price': price, 'Eq': equity, 'PnL': 0.0})
                    elif signal.action == TradeAction.SELL:
                        position = -1
                        entry_price = price
                        results.append({'Date': idx, 'Type': 'ENTRY_SHORT', 'Price': price, 'Eq': equity, 'PnL': 0.0})

        # Close final position
        if position != 0:
             last_price = df.iloc[-1]['Close']
             if position == 1:
                 pnl = (last_price - entry_price) / entry_price * equity
             else:
                 pnl = (entry_price - last_price) / entry_price * equity
             equity += pnl
             results.append({'Date': df.index[-1], 'Type': 'formatted', 'Price': last_price, 'Eq': equity, 'PnL': 0.0})
             
        return pd.DataFrame(results)

    def _mock_vision_history(self, df) -> VisionOutput:
        """
        Proxy vision bias using Price Structure.
        """
        last = df.iloc[-1]
        sma_50 = df['Close'].rolling(50).mean().iloc[-1]
        
        bias = MarketBias.NEUTRAL
        if last['Close'] > sma_50 * 1.02: bias = MarketBias.BULLISH
        elif last['Close'] < sma_50 * 0.98: bias = MarketBias.BEARISH
        
        return VisionOutput(
            market_bias=bias,
            breakout_prob=0.5,
            reversal_prob=0.2,
            momentum_score=0.6,
            patterns_detected=["Simulated_History"]
        )

    def calculate_metrics(self, trades_df: pd.DataFrame, initial_capital: float) -> dict:
        """
        Detailed Institutional Metrics.
        """
        if trades_df.empty:
            return {"Total Return": "0%", "Sharpe": 0.0, "Max DD": "0%", "Win Rate": "0%"}
            
        final_eq = trades_df.iloc[-1]['Eq']
        total_ret = (final_eq - initial_capital) / initial_capital
        
        # Max Drawdown
        equity_curve = trades_df['Eq'].values
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / running_max
        max_dd = np.max(drawdowns)
        
        # Win Rate (Filter Exits)
        exits = trades_df[trades_df['Type'].str.contains('EXIT')]
        if len(exits) > 0:
            wins = len(exits[exits['PnL'] > 0])
            win_rate = wins / len(exits)
        else:
            win_rate = 0.0
            
        # Simulated Sharpe (Annualized approx from trade returns)
        # Real sharpe needs daily returns
        if len(exits) > 1:
            rets = exits['PnL']
            sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() != 0 else 0
        else:
            sharpe = 0.0
            
        return {
            "Total Return": f"{total_ret:.1%}",
            "Sharpe": f"{sharpe:.2f}",
            "Max DD": f"{max_dd:.1%}",
            "Win Rate": f"{win_rate:.1%}",
            "Trades": len(exits)
        }
