"""
Step 33: HFT Order Flow Engine.
Simulates Level 2 Order Book data and calculates Order Flow Imbalance.
Essential for micro-scalping decisions (e.g., Don't buy into a Sell Wall).
"""
from dataclasses import dataclass
import random
import numpy as np

@dataclass
class OrderFlowState:
    imbalance: float   # -1.0 (Full Bearish) to 1.0 (Full Bullish)
    buy_vol: float     # Simulated Buy Volume impact
    sell_vol: float    # Simulated Sell Volume impact
    dominant_side: str # "BULLS" or "BEARS"
    whale_wall: str    # "NONE", "BID_WALL", "ASK_WALL"

class HFTEngine:
    """
    High-Frequency Trading Engine.
    Simulates L2 data dynamics.
    """
    
    def __init__(self):
        pass
        
    def analyze(self, current_price: float) -> OrderFlowState:
        """
        Simulate Order Book snapshot.
        """
        # Simulate Random Order Flow
        # In a real system, this would aggregate recent ticks or L2 snapshot
        
        # 1. Generate Buy/Sell Pressures
        base_vol = 1000.0
        buy_pressure = random.uniform(0.5, 1.5) * base_vol
        sell_pressure = random.uniform(0.5, 1.5) * base_vol
        
        # 2. Calculate Imbalance
        # Formula: (Buy - Sell) / (Buy + Sell)
        total_vol = buy_pressure + sell_pressure
        imbalance = 0.0
        if total_vol > 0:
            imbalance = (buy_pressure - sell_pressure) / total_vol
            
        # 3. Whale Wall Detection (Random Event)
        # 10% chance of a wall
        whale_wall = "NONE"
        rand_wall = random.random()
        if rand_wall < 0.05:
            whale_wall = "ASK_WALL" # Resistance
            sell_pressure *= 3.0 # Fake extra pressure
            imbalance = (buy_pressure - sell_pressure) / (buy_pressure + sell_pressure)
        elif rand_wall < 0.10:
            whale_wall = "BID_WALL" # Support
            buy_pressure *= 3.0
            imbalance = (buy_pressure - sell_pressure) / (buy_pressure + sell_pressure)
            
        # 4. Determine Side
        dom_side = "NEUTRAL"
        if imbalance > 0.2: dom_side = "BULLS"
        elif imbalance < -0.2: dom_side = "BEARS"
        
        return OrderFlowState(
            imbalance=round(imbalance, 2),
            buy_vol=round(buy_pressure, 0),
            sell_vol=round(sell_pressure, 0),
            dominant_side=dom_side,
            whale_wall=whale_wall
        )
