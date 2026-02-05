"""
Step 29: Institutional Money Manager.
Handles position sizing and risk limits.
"""
from src.core.config import risk_settings
from src.core.journal import Journal
from src.core.logger import SystemLogger

class MoneyManager:
    """
    Capital Allocation Engine.
    Strategies: Fixed, Percent, Kelly Criterion.
    """
    
    def __init__(self):
        self.journal = Journal()
        self.logger = SystemLogger()
        self.settings = risk_settings.copy()
        
    def update_settings(self, new_settings: dict):
        """Update risk parameters efficiently."""
        self.settings.update(new_settings)
        
    def check_daily_loss_limit(self) -> bool:
        """
        Check if we hit the Daily Max Loss.
        Returns False if trading should STOP.
        """
        metrics = self.journal.get_metrics()
        # Note: Ideally Journal needs a 'daily_pnl' method. 
        # For now, we assume 'pnl' in metrics is total, enabling simpler check if we reset journal daily
        # Or we scan history for today's trades.
        
        # Scan today's trades from history
        df = self.journal.get_history()
        if df.empty:
            return True
            
        # Filter for Today (Naive String Match)
        today = df['Timestamp'].astype(str).str.slice(0, 10).iloc[0] # assuming sorted
        today_trades = df[df['Timestamp'].str.startswith(today)]
        
        daily_pnl = today_trades['PnL'].sum()
        
        if daily_pnl <= -self.settings['max_daily_loss']:
            self.logger.warning(f"RISK: Daily Loss Limit Hit (${daily_pnl:.2f})")
            return False
            
        return True

    def get_trade_amount(self, confidence: float) -> float:
        """
        Calculate Position Size based on Strategy.
        """
        # Safety Check
        if not self.check_daily_loss_limit():
            return 0.0
            
        strategy = self.settings['strategy']
        base = self.settings['base_amount']
        
        amount = base
        
        if strategy == "Fixed":
            amount = base
            
        elif strategy == "Percent":
            # Mock Equity $1000 for relative check, or use base as 'Equity'
            # In real system, we need 'Account Balance' from API
            equity = 1000.0 # Placeholder
            pct = self.settings.get('percent_equity', 0.05)
            amount = equity * pct
            
        elif strategy == "Kelly":
            # Adaptive: Uses Journal Win Rate
            metrics = self.journal.get_metrics()
            win_rate = metrics.get('win_rate', 0.5)
            if win_rate < 0.51: # No edge
                win_rate = 0.51 # Floor to avoid divestment in simulation
                
            # Kelly Formula: f = p - (q / b)
            # b = odds (assuming 0.85 payout typical for binary)
            p = win_rate
            q = 1 - p
            b = 0.85
            
            f = p - (q / b)
            
            # Scale (Quarter Kelly for Safety)
            k_scale = self.settings.get('kelly_fraction', 0.25)
            f_sized = f * k_scale
            
            equity = 1000.0 # Placeholder
            amount = max(base, equity * f_sized)
            
        # Confidence Scaling (Optional Boost)
        if confidence > 0.85:
            amount *= 1.2
            
        return round(max(1.0, amount), 2) # Min $1
