"""
Step 31: Market Sentiment Engine.
Simulates fundamental analysis by analyzing news headlines.
"""
import random
from dataclasses import dataclass
from typing import List

@dataclass
class NewsItem:
    headline: str
    impact: float # -1.0 to 1.0
    source: str
    timestamp: str

@dataclass
class SentimentOutput:
    score: float # -1.0 to 1.0
    bias: str # BULLISH, BEARISH, NEUTRAL
    headlines: List[NewsItem]

class SentimentEngine:
    """
    Simulates a News Aggregator and Sentiment Analyzer (VADER-style logic).
    Since we don't have a real News API, we generate realistic scenarios.
    """
    
    def __init__(self):
        self.scenarios = [
            ("SEC approves new Bitcoin ETF application", 0.9, "Bloomberg"),
            ("Federal Reserve hints at rate cuts next month", 0.7, "Reuters"),
            ("Major crypto exchange suffers security breach", -0.8, "CoinDesk"),
            ("Inflation data comes in lower than expected", 0.6, "CNBC"),
            ("Tech stocks rally on strong earnings reports", 0.5, "Yahoo Finance"),
            ("Whale moves 10,000 BTC to exchange", -0.4, "WhaleAlert"),
            ("Market consolidating ahead of FOMC meeting", 0.1, "Analyst"),
            ("Global supply chain issues persist", -0.3, "Economist"),
            ("Bitcoin mining difficulty reaches all-time high", 0.4, "Blockchain.com"),
            ("Regulatory crackdown warnings in Eurozone", -0.6, "FT")
        ]
        
    def analyze(self, ticker: str = "BTC") -> SentimentOutput:
        """
        Generate a sentiment snapshot.
        In a real system, this would fetch from NewsAPI/Twitter.
        """
        # Simulate fetching 3 random active headlines
        news_batch = random.sample(self.scenarios, 3)
        
        items = []
        total_score = 0.0
        
        for head, impact, src in news_batch:
            # Add some noise to simulated impact
            noise = random.uniform(-0.1, 0.1)
            final_impact = max(-1.0, min(1.0, impact + noise))
            
            items.append(NewsItem(
                headline=head,
                impact=final_impact,
                source=src,
                timestamp="Just Now"
            ))
            total_score += final_impact
            
        # Average Score
        avg_score = total_score / 3.0
        
        # Determine Bias
        bias = "NEUTRAL"
        if avg_score > 0.3: bias = "BULLISH"
        elif avg_score < -0.3: bias = "BEARISH"
        
        return SentimentOutput(
            score=round(avg_score, 2),
            bias=bias,
            headlines=items
        )
