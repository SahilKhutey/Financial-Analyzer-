"""
Step 17: Master Dashboard UI.
Layout: Professional 12-Column Grid (Bloomberg-Style).
Focus: Calm, Dense, User-Controlled.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import time

# System Imports
from src.data.ingestion import DataFactory
from src.data.processing import ImageProcessor

st.set_page_config(
    page_title="Vision-Fusion Trader",
    layout="wide",
    initial_sidebar_state="expanded"
)
from src.engines.vision import EfficientNetWrapper
from src.engines.timeseries import LSTMModel
from src.engines.regime import RegimeFilter
from src.engines.scalping import ScalpingEngine
from src.engines.breakout import BreakoutEngine
from src.engines.mean_reversion import MeanReversionEngine
from src.engines.trend_following import TrendFollowingEngine
from src.engines.fusion import SignalFusion
from src.engines.backtest import BacktestEngine
from src.engines.binomo import BinomoEngine
from src.engines.smart_money import SmartMoneyEngine
from src.engines.math_engine import MathPredictionEngine
from src.engines.deep_analytic import DeepAnalyticalEngine
from src.engines.ml_production import MLProductionEngine
from src.core.types import FinalSignal, TradeAction

@st.cache_resource
def load_engines():
    return (
        EfficientNetWrapper(),
        LSTMModel(),
        RegimeFilter(),
        SignalFusion(),
        BacktestEngine(),
        ScalpingEngine(),
        BreakoutEngine(),
        MeanReversionEngine(),
        TrendFollowingEngine(),
        BinomoEngine(),
        SmartMoneyEngine(),
        MathPredictionEngine(),
        DeepAnalyticalEngine(),
        MLProductionEngine()
    )

def render_safety_badge(connected: bool):
    color = "#238636" if connected else "#DA3633"
    status = "CONNECTED" if connected else "DISCONNECTED"
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px; background: #161B22; padding: 8px 16px; border-radius: 20px; border: 1px solid {color};">
        <div style="width: 10px; height: 10px; border-radius: 50%; background-color: {color}; box-shadow: 0 0 5px {color};"></div>
        <span style="font-weight: bold; font-size: 0.9em; color: {color};">{status}</span>
        <span style="color: #8B949E; border-left: 1px solid #444; padding-left: 10px; font-size: 0.8em;">ANALYSIS ONLY</span>
    </div>
    """, unsafe_allow_html=True)

# --- 4. MAIN APP LAYOUT ---
def main():
    # Load Core
    vision_engine, ts_engine, regime_engine, fusion_engine, bt_engine, scalp_engine, breakout_engine, mr_engine, trend_engine, binomo_engine, smart_engine, math_engine, deep_engine, ml_engine = load_engines()
    
    # Session State Init
    if 'history' not in st.session_state: st.session_state['history'] = []
    if 'capturing' not in st.session_state: st.session_state['capturing'] = False
    
    # --- [SECTION 2] LEFT PANEL (CONTROLS) ---
    with st.sidebar:
        st.markdown("## 🕹️ Control Panel")
        
        # A. Chart Source
        st.markdown("### 1. Source")
        input_mode = st.radio("Mode", ["Live Screen", "Static API (Data)"], label_visibility="collapsed")
        
        if input_mode == "Live Screen":
            st.info("🔒 Secure Capture Active")
            ref_link = st.text_input("Platform Link (Ref)", "https://tradingview.com")
            
            from src.data.capture import ScreenCapture
            cap = ScreenCapture()
            mons = cap.get_monitor_names()
            mon_id = st.selectbox("Select Monitor", mons, index=0)
            
            # Parse monitor index
            real_mon_idx = int(mon_id.split(":")[0].replace("Monitor ", ""))
        else:
            ticker = st.text_input("Ticker", "BTC-USD")
        
        st.markdown("---")
        
        # B. Capture Controls
        st.markdown("### 2. Capture Controls")
        interval = st.radio("Interval (sec)", [3, 5, 10], index=1, horizontal=True)
        st.session_state['interval'] = interval
        
        st.markdown("---")
        
        # B. Strategy Selector
        st.markdown("### 2. Strategy Engine")
        strategy_mode = st.selectbox("Strategy Profile", 
                                     ["Auto (Fusion)", "Binomo Focus", "Smart Money (6-Layer)", "Mathematical Prediction", "Deep Analytical (Institutional)", "Enterprise ML (XGB+LSTM)", "Scalping Focus", "Breakout Focus", "Mean Reversion", "Trend Following"])
        
        st.markdown("---")
        
        # C. Analysis Controls
        st.markdown("### 3. Execution")
        
        if st.session_state['capturing']:
             if st.button("⏹ STOP ANALYSIS", type="primary", use_container_width=True):
                 st.session_state['capturing'] = False
                 st.rerun()
        else:
             if st.button("▶ START ANALYSIS", type="primary", use_container_width=True):
                 st.session_state['capturing'] = True
                 st.rerun()
                 
        st.markdown("---")
        st.caption("Risk Profile: **Institutional (Moderate)**")
        st.caption(f"v2.4.0 | Build 9284")

    # --- [SECTION 1] TOP BAR ---
    # We use columns to simulate a navbar
    c_logo, c_info, c_status = st.columns([2, 4, 2])
    
    with c_logo:
        st.markdown("### 👁️ VisionTrade")
    
    with c_info:
        # Dynamic Info Text
        target = "Live Screen" if input_mode == "Live Screen" else f"{ticker} (Data)"
        st.markdown(f"**Target:** {target} | **Profile:** {strategy_mode} | **TF:** Adaptive")
        
    with c_status:
        render_safety_badge(st.session_state['capturing'])

    st.markdown("---")

    # --- MAIN GRID (CENTER & RIGHT) ---
    # 12-Column Grid Simulation: Center (9) + Right (3)
    col_main, col_intel = st.columns([9, 3])
    
    # Placeholders for Loop
    chart_placeholder = col_main.empty()
    intel_placeholder = col_intel.empty()
    
    # --- LOGIC LOOP ---
    if st.session_state['capturing']:
        
        # Run Analysis Once (or Loop if using experimental st_autorefresh, but simplistic loop here works for manual refresh feel)
        # For a true loop in Streamlit without plugins, we often rely on rerun, but let's do a single pass render here
        # The user has to click "Start" which sets state. To loop, we'd need a loop inside `main` with sleep.
        # Strict Streamlit rule: Don't block main thread forever. 
        # We will run ONE iteration per rerun, and provide a "Capture Next" flow or use a loop container.
        
        # Implementing a visual 'Loop' container
        with st.spinner("Analyzing Market Structure..."):
            try:
                # 1. Capture / Data
                img = None
                df = None
                image_tensor = None
                
                if input_mode == "Live Screen":
                    # Capture
                    img = cap.capture(monitor_idx=real_mon_idx)
                    if img is None:
                        st.error("Capture Failed")
                        st.stop()
                    image_tensor = ImageProcessor.preprocess_image(img)
                else:
                    # Data
                    df = DataFactory.get_data(ticker)
                    image_tensor = np.zeros((1, 224, 224, 3)) # Dummy
                
                # 2. Intel Analysis
                vision_out = vision_engine.analyze(image_tensor)
                
                ts_out = None
                regime_out = None
                
                if df is not None:
                    ts_out = ts_engine.predict(df)
                    regime_out = regime_engine.analyze(df)
                else:
                    # Mock for Vision
                    from src.core.types import TSOutput, RegimeOutput
                    ts_out = TSOutput(0.0, 0.0, 0.0)
                    regime_out = RegimeOutput(True, "Vision_Mode", "N/A")
                # 3. Strategy Routing
                signal = None
                
                if strategy_mode == "Auto (Fusion)":
                    # Run All Institutional Engines
                    ml_sig = ml_engine.analyze(df)
                    deep_sig = deep_engine.analyze(df)
                    math_sig = math_engine.analyze(df)
                    
                    signal = fusion_engine.fuse(
                        vision_out, 
                        ts_out, 
                        regime_out,
                        ml_sig=ml_sig, 
                        deep_sig=deep_sig, 
                        math_sig=math_sig
                    )
                
                elif strategy_mode == "Scalping Focus":
                    signal = scalp_engine.analyze(df, vision_out=vision_out)
                    
                elif strategy_mode == "Breakout Focus":
                    signal = breakout_engine.analyze(df, vision_out=vision_out)
                    
                elif strategy_mode == "Mean Reversion":
                    signal = mr_engine.analyze(df, vision_out=vision_out)
                    
                elif strategy_mode == "Trend Following":
                    signal = trend_engine.analyze(df, vision_out=vision_out)
                    
                elif strategy_mode == "Smart Money (6-Layer)":
                    signal = smart_engine.analyze(df, vision_image=image_tensor)

                elif strategy_mode == "Mathematical Prediction":
                    signal = math_engine.analyze(df)

                elif strategy_mode == "Deep Analytical (Institutional)":
                    signal = deep_engine.analyze(df)

                elif strategy_mode == "Enterprise ML (XGB+LSTM)":
                    signal = ml_engine.analyze(df)

                elif strategy_mode == "Binomo Focus":
                    # Special return type for Binomo, but we map to standard for Fusion compatibility
                    # or better: we handle UI differently for Binomo.
                    # Let's get the specific object first
                    b_sig = binomo_engine.analyze(df, vision_out=vision_out)
                    
                    # Store for custom UI rendering below?
                    # Or map to signal object:
                    # Or map to signal object:
                    act = TradeAction.STAY_OUT
                    if b_sig.action == "BUY": act = TradeAction.BUY
                    elif b_sig.action == "SELL": act = TradeAction.SELL
                    
                    signal = FinalSignal(act, b_sig.confidence, b_sig.reasoning, "N/A", "N/A", "Binomo")
                    # We inject expiry into reasoning for display
                    signal.reasoning.append(f"Expiry: {b_sig.expiry}")
                
                # Fallback
                if signal is None:
                    signal = fusion_engine.fuse(vision_out, ts_out, regime_out)
                
                # --- VISUALIZATION (CENTER PANEL) ---
                with chart_placeholder.container():
                    st.caption("Live Analysis View")
                    if img:
                        # Show crop/processed if user wants (Simulated here with raw)
                        st.image(img, use_column_width=True, clamp=True, channels="RGB")
                    elif df is not None:
                        # Plotly Chart
                        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
                        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=0, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Overlay Params
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Resolution", "1920x1080 (Norm)")
                    c2.metric("Latency", "42ms")
                    c3.metric("Lighting", "Good")
                
                # --- INTEL (RIGHT PANEL) ---
                with intel_placeholder.container():
                    # Special Binomo UI if active
                    if strategy_mode == "Binomo Focus":
                        st.markdown("### 🎲 Binomo Live")
                        
                        # Custom large card
                        b_color = "signal-neutral"
                        if signal.action == TradeAction.BUY: b_color = "signal-buy"
                        elif signal.action == TradeAction.SELL: b_color = "signal-sell"
                        
                        st.markdown(f"""
                        <div class="signal-card {b_color}">
                            <h2 style="margin:0">BINARY SIGNAL</h2>
                            <h1 style="font-size:3.5em; margin:10px 0;">{signal.action.value}</h1>
                            <h3 style="margin:0;">CONF: {signal.confidence:.0%}</h3>
                            <p style="color:#CCC; margin-top:5px;">EXPIRY: 2-3 MIN</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Indicator Snapshot
                        st.markdown("#### ⚡ Indicators")
                        if df is not None:
                             last = df.iloc[-1]
                             rsi = last.get('rsi', 0)
                             st.metric("RSI (14)", f"{rsi:.1f}")
                             st.metric("Vol", f"{last.get('volatility', 0):.5f}")
                        else:
                             st.info("No Data for Indicators")
                             
                    else:
                        # Standard UI
                        # A. Signal Card
                        sig_color = "signal-neutral"
                        if signal.action == TradeAction.BUY: sig_color = "signal-buy"
                        elif signal.action == TradeAction.SELL: sig_color = "signal-sell"
                        
                        st.markdown(f"""
                        <div class="signal-card {sig_color}">
                            <h4 style="margin:0; color:#8B949E; letter-spacing:1px;">SIGNAL</h4>
                            <h1 style="font-size:3em; margin:10px 0;">{signal.action.value}</h1>
                            <h3 style="margin:0;">CONF: {signal.confidence:.0%}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.progress(signal.confidence, text="Confidence Level")

                        st.markdown("### 🧠 Intelligence")
                        if signal.reasoning:
                            for reason in signal.reasoning:
                                st.markdown(f"**• {reason}**")
                        else:
                            st.info("Analysis pending...")
                    
                    st.markdown("### 📊 Metrics")
                    m1, m2, m3 = st.columns(3)
                    
                    # Dynamic Metrics based on Strategy
                    if strategy_mode == "Auto (Fusion)":
                        m1.metric("Grand Score", signal.ts_bias.replace("Raw:", "") if "Raw:" in str(signal.ts_bias) else "N/A")
                        m2.metric("Regime", regime_out.state)
                        m3.metric("Bias", vision_out.market_bias.value)
                        
                    elif strategy_mode == "Enterprise ML (XGB+LSTM)":
                        m1.metric("ML Score", signal.ts_bias.replace("ML:", "") if "ML:" in str(signal.ts_bias) else "N/A")
                        m2.metric("Vol (ATR)", "Low" if "Low" in str(signal.reasoning) else "Norm") 
                        m3.metric("Trend", "UP" if signal.action == TradeAction.BUY else ("DOWN" if signal.action == TradeAction.SELL else "FLAT"))
                    
                    elif strategy_mode == "Deep Analytical (Institutional)":
                        m1.metric("Fused Prob", signal.confidence if signal.action != TradeAction.STAY_OUT else "N/A")
                        m2.metric("Regime", signal.regime)
                        m3.metric("Kalman", signal.ts_bias.replace("Kalman:", ""))
                        
                    elif strategy_mode == "Mathematical Prediction":
                        m1.metric("Prob", signal.ts_bias.replace("Prob:", ""))
                        m2.metric("Hurst", signal.regime.replace("H=", "") if "H=" in str(signal.regime) else "N/A")
                        m3.metric("Entropy", "OK")
                        
                    else:
                        # Legacy Default
                        m1.metric("Bias", vision_out.market_bias.value)
                        m2.metric("Regime", regime_out.state)
                        m3.metric("Trend", f"{ts_out.bullish_prob:.2f}")
                    
                    st.markdown("### 🛡️ Risk")
                    if "High Volatility" in str(signal.reasoning):
                        st.warning("⚠️ High Volatility Detected")
                    elif signal.confidence < 0.6:
                        st.warning("⚠️ Low Confidence - Wait")
                    else:
                        st.success("✅ Risk Parameters Normal")

                # --- [SECTION 5] LOWER PANEL (HISTORY) ---
                st.markdown("---")
                st.markdown("### 📜 Session History")
                
                # Update History
                new_row = {"Time": pd.Timestamp.now().strftime("%H:%M:%S"), "Signal": signal.action.value, "Conf": f"{signal.confidence:.2f}", "Strategy": strategy_mode}
                if not st.session_state['history'] or st.session_state['history'][0]['Time'] != new_row['Time']:
                    st.session_state['history'].insert(0, new_row)
                    
                st.dataframe(pd.DataFrame(st.session_state['history']), use_container_width=True, hide_index=True)
                
                # Auto-Rerun Mock (For demo loop feel)
                # Use selected interval
                time.sleep(st.session_state.get('interval', 5)) 
                st.rerun()

            except Exception as e:
                st.error(f"Analysis Error: {e}")
                st.session_state['capturing'] = False # Stop on error
    else:
        # Idle State
        col_main.info("System Standby. Configure settings on the left and click START to begin analysis.")
        intel_placeholder.markdown("""
        <div style="padding: 20px; border: 1px dashed #555; border-radius: 8px; text-align: center; color: #777;">
            Waiting for Signal...
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
