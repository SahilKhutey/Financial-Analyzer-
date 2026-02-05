"""
Step 25: Trade Journal Core.
Handles persistent logging of trades to CSV and calculates performance metrics.
"""
import os
import pandas as pd
from datetime import datetime
from src.core.types import TradeAction

JOURNAL_PATH = "user_data/trade_journal.csv"
TRAINING_DATA_PATH = "user_data/training_data.csv"

class Journal:
    """
    Singleton-style Journal Manager.
    """
    
    def __init__(self):
        self._ensure_file()
        
    def _ensure_file(self):
        directory = os.path.dirname(JOURNAL_PATH)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        if not os.path.exists(JOURNAL_PATH):
            df = pd.DataFrame(columns=[
                "Timestamp", "Asset", "Action", "Amount", "Strategy", "Reason", "Result", "PnL"
            ])
            df.to_csv(JOURNAL_PATH, index=False)

        if not os.path.exists(TRAINING_DATA_PATH):
             # Header for ML Features
             # We assume a standard set, but flexible columns are better for CSV
             df = pd.DataFrame(columns=["Timestamp", "Action", "Result"] + ["log_ret", "vol", "mom", "rsi", "vol_atr"])
             df.to_csv(TRAINING_DATA_PATH, index=False)
            
    def log_trade(self, asset: str, action: str, amount: float, strategy: str, reason: str, features: dict = None):
        """
        Log a new trade execution.
        Result is initially 'PENDING'.
        """
        if action == "STAY_OUT":
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        new_row = {
            "Timestamp": timestamp,
            "Asset": asset,
            "Action": action,
            "Amount": amount,
            "Strategy": strategy,
            "Reason": str(reason),
            "Result": "PENDING", # Needs manual or future auto-update
            "PnL": 0.0
        }
        
        # Append to Journal
        try:
            df = pd.read_csv(JOURNAL_PATH)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(JOURNAL_PATH, index=False)
            print(f"📝 Trade Logged: {action} on {asset}")
        except Exception as e:
            print(f"❌ Journal Log Error: {e}")
            
        # Append to Training Data (if features provided)
        if features:
            try:
                # Merge basic info + features
                train_row = {"Timestamp": timestamp, "Action": action, "Result": "PENDING"}
                train_row.update(features)
                
                t_df = pd.read_csv(TRAINING_DATA_PATH)
                t_df = pd.concat([t_df, pd.DataFrame([train_row])], ignore_index=True)
                t_df.to_csv(TRAINING_DATA_PATH, index=False)
            except Exception as e:
                print(f"❌ Training Data Log Error: {e}")

    def get_history(self) -> pd.DataFrame:
        """Return full trade history."""
        if os.path.exists(JOURNAL_PATH):
            return pd.read_csv(JOURNAL_PATH).sort_values(by="Timestamp", ascending=False)
        return pd.DataFrame()

    def get_metrics(self) -> dict:
        """Calculate Win Rate, Total PnL, etc."""
        df = self.get_history()
        if df.empty:
            return {"total": 0, "win_rate": 0.0, "pnl": 0.0}
            
        # Filter for completed trades
        completed = df[df["Result"].isin(["WIN", "LOSS"])]
        
        total = len(df)
        wins = len(completed[completed["Result"] == "WIN"])
        losses = len(completed[completed["Result"] == "LOSS"])
        
        completed_count = wins + losses
        win_rate = (wins / completed_count) if completed_count > 0 else 0.0
        
        # PnL (Mock Calculation for PENDING if needed, but strict PnL usually sums 'PnL' col)
        total_pnl = df["PnL"].sum()
        
        return {
            "total": total,
            "completed": completed_count,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "pnl": total_pnl
        }
