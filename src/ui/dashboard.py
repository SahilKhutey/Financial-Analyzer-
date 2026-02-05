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
from src.engines.yolo_logic import YOLOEngine
from src.engines.explanation import ExplanationEngine
from src.engines.optimizer import OptimizerEngine
from src.core.types import FinalSignal, TradeAction
from src.core.journal import Journal
from src.core.logger import SystemLogger
from src.core.profiler import Profiler, LatencyStats
from src.core.money_manager import MoneyManager
from src.core.money_manager import MoneyManager
from src.engines.trainer import ModelTrainer
from src.engines.sentiment import SentimentEngine
from src.engines.macro import MacroEngine
from src.engines.hft import HFTEngine

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
        MLProductionEngine(),
        ExplanationEngine(),
        OptimizerEngine(),
        ModelTrainer(),
        SentimentEngine(),
        MacroEngine(),
        HFTEngine()
    )

def render_safety_badge(status: str, color_code: str):
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px; background: #161B22; padding: 8px 16px; border-radius: 20px; border: 1px solid {color_code};">
        <div style="width: 10px; height: 10px; border-radius: 50%; background-color: {color_code}; box-shadow: 0 0 5px {color_code};"></div>
        <span style="font-weight: bold; font-size: 0.9em; color: {color_code};">{status}</span>
        <span style="color: #8B949E; border-left: 1px solid #444; padding-left: 10px; font-size: 0.8em;">SYSTEM ACTIVE</span>
    </div>
    """, unsafe_allow_html=True)

# --- 4. MAIN APP LAYOUT ---
def main():
    # Load Core
    vision_engine, ts_engine, regime_engine, fusion_engine, bt_engine, scalp_engine, breakout_engine, mr_engine, trend_engine, binomo_engine, smart_engine, math_engine, deep_engine, ml_engine, yolo_engine, expl_engine, opt_engine, trainer_engine, sent_engine, macro_engine, hft_engine = load_engines()
    
    # Session State Init
    if 'history' not in st.session_state: st.session_state['history'] = []
    if 'capturing' not in st.session_state: st.session_state['capturing'] = False
    
    # Init Logger
    logger = SystemLogger()
    
    # Init Profiler Stats
    if 'latency_stats' not in st.session_state: 
        st.session_state['latency_stats'] = LatencyStats()
    stats = st.session_state['latency_stats']
    
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
                                     ["Auto (Fusion)", "Vision (YOLOv8)", "Binomo Focus", "Smart Money (6-Layer)", "Mathematical Prediction", "Deep Analytical (Institutional)", "Enterprise ML (XGB+LSTM)", "Scalping Focus", "Breakout Focus", "Mean Reversion", "Trend Following"])
        
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
        
        # D. Risk Management
        with st.expander("💰 Risk & Money"):
            # A singleton init trick or re-access engine
            mm_instance = binomo_engine.money_manager # Access engine's manager
            
            risk_strat = st.selectbox("Sizing Model", ["Fixed", "Percent", "Kelly"], index=0)
            
            base_amt = st.number_input("Base Amount ($)", value=10.0, step=1.0)
            
            max_daily = st.number_input("Daily Loss Limit ($)", value=50.0, step=10.0)
            
            if st.button("Update Risk"):
                new_set = {
                    'strategy': risk_strat,
                    'base_amount': base_amt,
                    'max_daily_loss': max_daily
                }
                mm_instance.update_settings(new_set)
                st.toast("Risk Settings Updated!")
        
        st.markdown("---")
        st.caption("Risk Profile: **Institutional (Moderate)**")
        st.caption(f"v2.8.0 | Build 9301")
        
        # E. Sentiment Feed (Mini)
        with st.expander("📰 Market News (Live)", expanded=False):
            # We can't access 'sent_out' easily here as it's computed inside the loop
            # But we can call it once for display or use session state if we wanted sync
            # For UI responsiveness, let's just generate a snapshot here
            snap = sent_engine.analyze("BTC")
            
            s_color = "red" if snap.score < 0 else "green"
            st.markdown(f"**Sentiment Score:** :{s_color}[{snap.score}]")
            
            for item in snap.headlines:
                icon = "📉" if item.impact < 0 else "📈"
                st.markdown(f"{icon} **{item.source}**: {item.headline}")

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
        # Determine Status
        status_text = "READY"
        status_color = "#8B949E" # Gray
        
        if st.session_state['capturing']:
            if input_mode == "Live Screen":
                status_text = "VISION ONLY"
                status_color = "#D29922" # Amber
            else:
                status_text = "DATA FEED"
                status_color = "#238636" # Green
                
        render_safety_badge(status_text, status_color)

    st.markdown("---")
    
    # --- GLOBAL MACRO STRIP ---
    # Run Macro Analysis Once
    macro_state = macro_engine.analyze()
    
    # Render Strip - Bloomberg Style
    m_cols = st.columns([1, 1, 1, 2])
    with m_cols[0]:
        st.metric("DXY (Dollar)", f"{macro_state.dxy}", delta=f"{macro_state.dxy - 104.0:.2f}", delta_color="inverse")
    with m_cols[1]:
        v_col = "normal" if macro_state.vix < 20 else "inverse"
        st.metric("VIX (Fear)", f"{macro_state.vix}", delta=f"{macro_state.vix - 18.0:.2f}", delta_color=v_col)
    with m_cols[2]:
        st.metric("US10Y", f"{macro_state.us10y}%")
    with m_cols[3]:
        # Regime Badge
        r_color = "green" if macro_state.regime == "RISK_ON" else ("red" if "RISK_OFF" in macro_state.regime or "RECESSION" in macro_state.regime else "gray")
        st.markdown(f"#### 🌍 Regime: :{r_color}[{macro_state.regime}]")
        st.caption(f"Size Multiplier: {macro_state.multiplier}x")
        
    st.markdown("---")

    # --- TABS ---
    tab_term, tab_perf, tab_learn = st.tabs(["🖥️ Live Terminal", "📊 Performance", "🧠 Model Lab"])
    
    # --- TAB 1: LIVE TERMINAL ---
    with tab_term:
        # 12-Column Grid Simulation: Center (9) + Right (3)
        col_main, col_intel = st.columns([9, 3])
        
        # Placeholders for Loop
        chart_placeholder = col_main.empty()
        
        # HFT Strip (Bottom of Chart)
        hft_placeholder = col_main.empty()
        
        intel_placeholder = col_intel.empty()
    
    # --- TAB 2: PERFORMANCE ---
    with tab_perf:
        st.markdown("### 📈 Trading Performance")
        journal = Journal()
        metrics = journal.get_metrics()
        
        # 1. Top Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Signals", metrics['total'])
        m2.metric("Win Rate", f"{metrics['win_rate']:.1%}")
        m3.metric("Profit Factor", "1.5 (Est)") # Mock for now until outcomes are tracked
        
        pnl = metrics['pnl']
        pnl_color = "normal"
        if pnl > 0: pnl_color = "off" # Streamlit metric delta color logic is auto
        m4.metric("Net PnL", f"${pnl:.2f}", delta=f"{pnl:.2f}")

        st.markdown("---")
        
        # 2. Charts & Data
        c_chart, c_table = st.columns([2, 1])
        
        with c_chart:
            st.markdown("#### 💸 Equity Curve (Simulated)")
            df_hist = journal.get_history()
            if not df_hist.empty:
                # Cumulative Sum of PnL over time
                # We need to ensure PnL is numeric
                # In current journal, PnL is 0 until updated
                # Let's mock a cumulative count of 'Amount' for visual if PnL is 0
                
                # Mock Curve: Just count of trades for now
                visits = df_hist['Action'].value_counts()
                st.bar_chart(visits, color="#238636")
            else:
                st.info("No Data yet.")
                
        
        with c_table:
            st.markdown("#### 📝 Trade Log")
            st.dataframe(journal.get_history(), use_container_width=True, height=400)
            
        st.markdown("---")
        
        # 3. AUTO-OPTIMIZER
        st.markdown("### 🧠 Self-Optimization")
        with st.expander("⚙️ Run Auto-Tuner (Simulation)"):
            st.info("System will simulate 5 different fusion configurations on recent data to find the best fit.")
            if st.button("🚀 Run Grid Search Optimizer"):
                if df is not None:
                    with st.spinner("Running Grid Search on 500 candles..."):
                         results = opt_engine.run_optimization(df, top_n=3)
                         
                         if results:
                             best = results[0]
                             st.success(f"Opimization Complete! Best Config: **{best['name']}**")
                             
                             # Display Comparison
                             res_df = pd.DataFrame(results)[['name', 'score', 'sharpe', 'win_rate']]
                             st.dataframe(res_df, use_container_width=True)
                             
                             st.markdown(f"**Recommended Weights**: {best['weights']}")
                             
                             if st.button("✅ Apply Best Weights"):
                                 # In a real app, we would write to config.py or update session state
                                 # For this runtime, we update the live engine
                                 fusion_engine.set_weights(best['weights'])
                                 st.toast(f"Updated Fusion Engine to: {best['name']}")
                         else:
                             st.warning("Not enough data to optimize.")
                else:
                    st.error("No Data Available (Switch Input Source)")
                    
    # --- [SECTION 5] MODEL LAB TAB ---
    with tab_learn:
        st.subheader("🧠 Continuous Learning Pipeline")
        st.caption("Train the XGBoost model on your real trading data.")
        
        # Stats
        stats = trainer_engine.get_dataset_stats()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Dataset Size", f"{stats['count']} samples")
        c2.metric("Wins Recorded", stats['wins'])
        c3.metric("Losses Recorded", stats['losses'])
        
        st.markdown("---")
        
        if st.button("🚀 Retrain Model (XGBoost v2)", use_container_width=True):
            with st.spinner("Training Model on collected data..."):
                res = trainer_engine.train_model()
                if "Success" in res:
                    st.success(res)
                    st.balloons()
                else:
                    st.error(res)
                    
        with st.expander("ℹ️ How it works"):
            st.markdown("""
            1. **Collect**: Every trade captures market features (RSI, Volatility, etc.).
            2. **Label**: The system waits for the trade outcome (WIN/LOSS).
            3. **Train**: Clicking the button feeds this 'Gold Data' into XGBoost.
            4. **Deploy**: The new model file overrides the old one instantly.
            """)
    
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
                with Profiler("Vision", stats):
                    vision_out = vision_engine.analyze(image_tensor)
                
                ts_out = None
                regime_out = None
                
                if df is not None:
                     with Profiler("TimeSeries", stats):
                        ts_out = ts_engine.predict(df)
                     with Profiler("Regime", stats):
                        regime_out = regime_engine.analyze(df, macro_state=macro_state) # Pass Macro
                        
                # 3. Sentiment Analysis
                sent_out = None
                with Profiler("Sentiment", stats):
                    # For demo, we use a fixed ticker or derive from UI
                    t_sym = ticker if input_mode != "Live Screen" else "BTC"
                    sent_out = sent_engine.analyze(t_sym)
                
                # 4. Strategy Routing
                signal = None
                
                # Pre-calculate data signals if data exists
                ml_sig = None
                deep_sig = None
                math_sig = None
                
                if df is not None:
                    with Profiler("ML Engines", stats):
                        ml_sig = ml_engine.analyze(df)
                        deep_sig = deep_engine.analyze(df)
                        math_sig = math_engine.analyze(df)
                
                with Profiler("Fusion", stats):
                    if strategy_mode == "Auto (Fusion)":
                        signal = fusion_engine.fuse(
                            vision_out, 
                            ts_out, 
                            regime_out,
                            sentiment=sent_out,
                            ml_sig=ml_sig, 
                            deep_sig=deep_sig, 
                            math_sig=math_sig
                        )
                    
                    elif strategy_mode == "Scalping Focus":
                        signal = scalp_engine.analyze(df, vision_out=vision_out, hft_out=hft_out)
                        
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
    
                    elif strategy_mode == "Vision (YOLOv8)":
                        # 1. Vision Logic
                        signal = yolo_engine.analyze(image_tensor)
                        
                        # 2. Draw Detections
                        annotated_img = None
                        if img is not None:
                            np_img = np.array(img.convert('RGB')) # RGB
                            # no BGR conversion needed for yolo engine helper if it just draws colors
                            np_img_annotated = yolo_engine.get_annotated_image(np_img)
                            annotated_img = Image.fromarray(np_img_annotated)
                        else:
                            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                            np_img_annotated = yolo_engine.get_annotated_image(dummy)
                            annotated_img = Image.fromarray(np_img_annotated)
                            
                        with chart_placeholder.container():
                            st.caption("YOLO Object Detection View")
                            st.image(annotated_img, use_container_width=True, channels="RGB")
                            
                            # --- CONTROLS ---
                            c1, c2 = st.columns(2)
                            with c1:
                                auto_trade = st.toggle("⚡ Auto-Trade (Binomo)", value=False)
                            with c2:
                                if st.button("📸 Save for Training", use_container_width=True):
                                    if img:
                                        # Save raw PIL -> Numpy RGB
                                        # Engine handles RGB->BGR save
                                        path = yolo_engine.save_snapshot(np.array(img))
                                        st.toast(f"Saved: {path}")
                                        logger.info(f"Dataset Snapshot Saved: {path}")
                                    else:
                                        st.warning("No Image to Save")
                            
                            # --- EXECUTION BRIDGE ---
                            if auto_trade and signal.action != TradeAction.STAY_OUT:
                                res = binomo_engine.execute_trade(signal)
                                if "Placed" in res:
                                    st.toast(res, icon="⚡")
                                    logger.info(f"Auto-Trade Executed: {res}")
    
    
                    elif strategy_mode == "Binomo Focus":
                        # 1. Calculate Institutional "Fusion" Signal (Silent)
                        # Uses pre-calcs above (ml_sig etc)
                        
                        fusion_sig = fusion_engine.fuse(
                            vision_out, 
                            ts_out, 
                            regime_out,
                            ml_sig=ml_sig, 
                            deep_sig=deep_sig, 
                            math_sig=math_sig
                        )
                        
                        # 2. Pass to Binomo Execution Engine
                        # Now it aggregates Fusion + Vision + Heuristics
                        b_sig = binomo_engine.analyze(df, vision_out=vision_out, fusion_sig=fusion_sig)
                        
                        # 3. Map to standard FinalSignal for UI compatibility
                        act = TradeAction.STAY_OUT
                        if b_sig.action == "BUY": act = TradeAction.BUY
                        elif b_sig.action == "SELL": act = TradeAction.SELL
                        
                        signal = FinalSignal(act, b_sig.confidence, b_sig.reasoning, "N/A", "N/A", "Binomo")
                        # We inject expiry into reasoning for display
                        signal.reasoning.append(f"⏱️ Expiry: {b_sig.expiry}")
                    
                    # Fallback
                    if signal is None:
                        signal = fusion_engine.fuse(vision_out, ts_out, regime_out)
                
                # Log Signal
                if signal.action != TradeAction.STAY_OUT:
                    logger.info(f"SIGNAL GEN: {signal.action.value} | Conf: {signal.confidence:.2f} | Mode: {strategy_mode}")
                
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
                    lat = stats.timings.get("Fusion", 0) + stats.timings.get("Vision", 0) + stats.timings.get("ML Engines", 0)
                    c2.metric("Latency", f"{lat:.0f}ms")
                    c3.metric("Lighting", "Good")
                
                # --- HFT TAPE ---
                with hft_placeholder.container():
                    # Run HFT
                    hft_out = hft_engine.analyze(0.0)
                    
                    st.markdown("#### 🌊 Institutional Order Flow (Level 2)")
                    
                    # Imbalance Bar
                    # Visualize -1 to 1 as Red to Green Bar
                    imb_pct = (hft_out.imbalance + 1) / 2 # 0 to 1
                    
                    bar_color = "linear-gradient(90deg, #DA3633, #238636)"
                    marker_pos = imb_pct * 100
                    
                    st.markdown(f"""
                    <div style="width: 100%; height: 20px; background: #30363D; border-radius: 10px; position: relative;">
                         <div style="width: 100%; height: 100%; background: {bar_color}; border-radius: 10px; opacity: 0.5;"></div>
                         <div style="position: absolute; left: {marker_pos}%; top: -5px; width: 4px; height: 30px; background: #FFF; box-shadow: 0 0 10px white;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.8em; margin-top: 5px;">
                        <span style="color: #DA3633">Total BEARS: {hft_out.sell_vol:.0f}</span>
                        <span style="font-weight: bold;">{hft_out.dominant_side} CONTROL ({hft_out.imbalance:+.2f})</span>
                        <span style="color: #238636">Total BULLS: {hft_out.buy_vol:.0f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if hft_out.whale_wall != "NONE":
                        wall_color = "#DA3633" if "ASK" in hft_out.whale_wall else "#238636"
                        st.markdown(f"""
                        <div style="background: {wall_color}33; border: 1px solid {wall_color}; padding: 5px; border-radius: 5px; text-align: center; margin-top: 5px;">
                            🐋 <b>WHALE DETECTED:</b> {hft_out.whale_wall}
                        </div>
                        """, unsafe_allow_html=True)
                    
                # --- INTEL (RIGHT PANEL) ---
                # --- INTEL (RIGHT PANEL) ---
                with intel_placeholder.container():
                    # 1. LIVE REGIME BADGE
                    regime_color = "#238636" if regime_out and regime_out.is_safe else "#DA3633"
                    regime_txt = regime_out.state if regime_out else "UNCERTAIN"
                    st.markdown(f"""
                    <div style="text-align: right; font-size: 0.8em; color: #8B949E; margin-bottom: 5px;">
                        MARKET REGIME: <span style="color: {regime_color}; font-weight: bold;">{regime_txt}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # 2. CONFIDENCE METER (Gauge)
                    # Circular feel using CSS or simple Progress with color
                    conf_val = signal.confidence
                    meter_color = "#238636" if conf_val > 0.7 else ("#D29922" if conf_val > 0.5 else "#8B949E")
                    
                    st.markdown(f"""
                    <div style="background: #161B22; padding: 15px; border-radius: 10px; border: 1px solid #30363D; text-align: center;">
                        <h4 style="margin:0; color: #8B949E; font-size: 0.8em;">CONFIDENCE SCORE</h4>
                        <div style="font-size: 2.5em; font-weight: 800; color: {meter_color};">{conf_val:.0%}</div>
                        <div style="height: 6px; width: 100%; background: #30363D; border-radius: 3px; margin-top: 5px;">
                            <div style="height: 100%; width: {conf_val*100}%; background: {meter_color}; border-radius: 3px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("") # Spacer

                    # 3. SIGNAL CARD
                    sig_color = "signal-neutral"
                    if signal.action == TradeAction.BUY: sig_color = "signal-buy"
                    elif signal.action == TradeAction.SELL: sig_color = "signal-sell"
                    
                    st.markdown(f"""
                    <div class="signal-card {sig_color}">
                        <h5 style="margin:0; color:#EEE; letter-spacing:1px; text-transform:uppercase;">AI DECISION</h5>
                        <h1 style="font-size:3.2em; margin:5px 0;">{signal.action.value}</h1>
                    </div>
                    """, unsafe_allow_html=True)

                    # 4. WHY THIS SIGNAL? (Checklist)
                    st.markdown("### ✅ Why this signal?")
                    checks = expl_engine.generate_checklist(signal)
                    for c in checks:
                        icon = "✅" if c['passed'] else "⬜"
                        st.markdown(f"{icon} {c['label']}")

                    st.markdown("---")

                    # 5. MODEL VOTES
                    st.markdown("### 🗳️ Model Votes")
                    if signal.model_votes:
                        for model, vote in signal.model_votes.items():
                            v_color = "#238636" if vote == "BUY" else ("#DA3633" if vote == "SELL" else "#8B949E")
                            st.markdown(f"""
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 0.9em;">
                                <span>{model}</span>
                                <span style="font-weight: bold; color: {v_color};">{vote}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.caption("No voting data available.")

                    st.markdown("---")
                    
                    # 6. AI ANALYST (Explanation)
                    st.markdown("### 🤖 Market Analyst")
                    narrative = expl_engine.get_explanation(signal)
                    st.info(narrative)
                    
                    with st.expander("🔬 View AI Prompt"):
                        st.code(expl_engine.generate_ai_prompt(signal), language="text")
                    
                    # 7. Session Stats
                    total_sigs = len(st.session_state['history'])
                    st.caption(f"Session Activity: **{total_sigs} Signals** Generated")

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

                with st.expander("⚡ System Health (Latency)"):
                    cols = st.columns(len(stats.timings)) if stats.timings else [st.empty()]
                    for i, (k, v) in enumerate(stats.timings.items()):
                         if i < len(cols):
                             cols[i].metric(k, f"{v:.1f}ms")
                             
                # ... loop end ...
 
            except Exception as e:
                logger.error(f"CRITICAL UI CRASH: {e}")
                st.error(f"System Error: {e}")
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
