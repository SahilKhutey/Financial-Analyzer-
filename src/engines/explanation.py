"""
Step 23: Dashboard Explanation Engine.
Bridging the gap between Signal Data and Human Understanding.
Features:
- "Why this signal?" Checklist Generator.
- AI Prompt Construction (for LLM Integration).
- Narrative Generation.
"""

from typing import List, Dict, Optional
from src.core.types import FinalSignal, TradeAction

class ExplanationEngine:
    """
    Translates technical signals into human-readable narratives and checklists.
    """
    
    def generate_checklist(self, signal: FinalSignal) -> List[dict]:
        """
        Returns a list of boolean checks for the UI.
        Format: [{'label': 'EMA Trend', 'passed': True}, ...]
        """
        checks = []
        reason_str = str(signal.reasoning).lower()
        
        # 1. Trend Logic
        checks.append({
            "label": "Bullish Trend",
            "passed": signal.action == TradeAction.BUY or "bull" in reason_str
        })
        
        # 2. Volatility
        checks.append({
            "label": "Low Volatility (Safe)",
            "passed": "high vol" not in reason_str and "crash" not in reason_str
        })
        
        # 3. Pattern / Confluence
        checks.append({
            "label": "Setup Pattern",
            "passed": any(x in reason_str for x in ["engulfing", "pinbar", "breakout", "structure"])
        })
        
        # 4. Confidence
        checks.append({
            "label": "High Confidence (>70%)",
            "passed": signal.confidence > 0.7
        })
        
        return checks

    def generate_ai_prompt(self, signal: FinalSignal, ticker: str = "Market") -> str:
        """
        Constructs the Prompt for the AI Analyst.
        This is what we WOULD send to GPT-4.
        """
        prompt = f"""
You are a senior trading analyst.
Explain this signal for {ticker} using price action, indicators, and risk logic.
Avoid guarantees. Use a professional, institutional tone.

SIGNAL: {signal.action.value}
CONFIDENCE: {signal.confidence:.0%}
REASONS: {signal.reasoning}
REGIME: {signal.regime}
VOTES: {signal.model_votes}
"""
        return prompt.strip()

    def get_explanation(self, signal: FinalSignal) -> str:
        """
        Returns a narrative explanation. 
        Currently Rule-Based, but structured like an AI response.
        """
        if signal.action == TradeAction.STAY_OUT:
            return f"**Analyst:** Market conditions are currently indifferent. Confidence is low ({signal.confidence:.0%}). We are waiting for a clearer setup or regime shift."
            
        direction = "UP" if signal.action == TradeAction.BUY else "DOWN"
        tone = "Bullish" if signal.action == TradeAction.BUY else "Bearish"
        
        narrative = f"""
**Analyst Note:**
We have identified a high-probability **{tone}** setup. 
The technical confluence allows for a **{signal.action.value}** bias.

**Key Drivers:**
*   **Momentum:** Directional probability is pointing {direction}.
*   **Structure:** {signal.regime} state supports this move.
*   **Consensus:** Multiple models (See Votes) align on this trajectory.

**Risk Note:**
Confidence is **{signal.confidence:.0%}**. 
Ensure proper position sizing as volatility remains dynamic.
"""
        return narrative.strip()
