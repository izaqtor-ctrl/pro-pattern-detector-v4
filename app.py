import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings

# Try to import yfinance with error handling
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Pro Pattern Detector v3.0", 
    layout="wide"
)

def create_demo_data(ticker, period):
    """Create realistic demo data when yfinance is not available"""
    days_map = {"1y": 252, "6mo": 126, "3mo": 63, "1mo": 22}
    days = days_map.get(period, 63)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(hash(ticker) % 2147483647)
    base_price = 150 + (hash(ticker) % 100)
    
    returns = np.random.normal(0.001, 0.02, days)
    returns[0] = 0
    
    close_prices = base_price * np.cumprod(1 + returns)
    
    high_mult = 1 + np.abs(np.random.normal(0, 0.01, days))
    low_mult = 1 - np.abs(np.random.normal(0, 0.01, days))
    open_mult = 1 + np.random.normal(0, 0.005, days)
    
    data = pd.DataFrame({
        'Open': close_prices * open_mult,
        'High': close_prices * high_mult,
        'Low': close_prices * low_mult,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 5000000, days)
    }, index=dates)
    
    data['High'] = np.maximum.reduce([data['Open'], data['High'], data['Low'], data['Close']])
    data['Low'] = np.minimum.reduce([data['Open'], data['High'], data['Low'], data['Close']])
    
    return data

def get_stock_data(ticker, period):
    """Fetch stock data with fallback to demo data"""
    if not YFINANCE_AVAILABLE:
        st.info(f"Using demo data for {ticker}")
        return create_demo_data(ticker, period)
    
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if len(data) == 0:
            st.warning(f"No data for {ticker}, using demo data")
            return create_demo_data(ticker, period)
        return data
    except Exception as e:
        st.warning(f"Error fetching {ticker}, using demo data")
        return create_demo_data(ticker, period)

def calculate_rsi(data, period=14):
    """Calculate RSI"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = data['Close'].ewm(span=fast).mean()
    ema_slow = data['Close'].ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def detect_flat_top(data, macd_line, signal_line, histogram):
    """Detect flat top: ASCENSION → DESCENSION → HIGHER LOWS"""
    confidence = 0
    pattern_info = {}
    
    if len(data) < 50:
        return confidence, pattern_info
    
    # STEP 1: Initial ascension (10%+ rise)
    ascent_start = min(45, len(data) - 15)
    ascent_end = 25
    
    start_price = data['Close'].iloc[-ascent_start]
    peak_price = data['High'].iloc[-ascent_start:-ascent_end].max()
    initial_gain = (peak_price - start_price) / start_price
    
    if initial_gain < 0.10:
        return confidence, pattern_info
    
    confidence += 25
    pattern_info['initial_ascension'] = f"{initial_gain*100:.1f}%"
    
    # STEP 2: Descension with lower highs
    descent_data = data.iloc[-ascent_end:-10]
    descent_low = descent_data['Low'].min()
    pullback = (peak_price - descent_low) / peak_price
    
    if pullback < 0.08:
        return confidence, pattern_info
    
    descent_highs = descent_data['High'].rolling(3, center=True).max().dropna()
    if len(descent_highs) >= 2:
        if descent_highs.iloc[-1] < descent_highs.iloc[0] * 0.97:
            confidence += 20
            pattern_info['descending_highs'] = True
    
    # STEP 3: Current higher lows
    current_lows = data.tail(15)['Low'].rolling(3, center=True).min().dropna()
    if len(current_lows) >= 3:
        if current_lows.iloc[-1] > current_lows.iloc[0] * 1.01:
            confidence += 25
            pattern_info['higher_lows'] = True
    
    # STEP 4: Flat resistance
    resistance_level = peak_price
    touches = sum(1 for h in data['High'].tail(20) if h >= resistance_level * 0.98)
    if touches >= 2:
        confidence += 15
        pattern_info['resistance_level'] = resistance_level
        pattern_info['resistance_touches'] = touches
    
    # STEP 5: Recency check
    current_price = data['Close'].iloc[-1]
    days_old = next((i for i in range(1, 11) if data['High'].iloc[-i] >= resistance_level * 0.98), 11)
    
    if days_old > 8:
        return confidence * 0.5, {**pattern_info, 'pattern_stale': True, 'days_old': days_old}
    
    if current_price < descent_low * 0.95:
        return 0, {'pattern_broken': True, 'break_reason': 'Below support'}
    
    # Technical confirmation
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 10
        pattern_info['macd_bullish'] = True
    
    return confidence, pattern_info

def detect_bull_flag(data, macd_line, signal_line, histogram):
    """Detect bull flag with recency validation"""
    confidence = 0
    pattern_info = {}
    
    if len(data) < 30:
        return confidence, pattern_info
    
    # Recent flagpole
    flagpole_start = min(25, len(data) - 10)
    flagpole_end = 15
    
    start_price = data['Close'].iloc[-flagpole_start]
    peak_price = data['High'].iloc[-flagpole_start:-flagpole_end].max()
    flagpole_gain = (peak_price - start_price) / start_price
    
    if flagpole_gain < 0.08:
        return confidence, pattern_info
    
    confidence += 25
    pattern_info['flagpole_gain'] = f"{flagpole_gain*100:.1f}%"
    
    # Flag pullback
    flag_data = data.tail(15)
    flag_start = data['Close'].iloc[-flagpole_end]
    current_price = data['Close'].iloc[-1]
    
    pullback = (current_price - flag_start) / flag_start
    if -0.15 <= pullback <= 0.05:
        confidence += 20
        pattern_info['flag_pullback'] = f"{pullback*100:.1f}%"
        pattern_info['healthy_pullback'] = True
    
    # Invalidation checks
    flag_low = flag_data['Low'].min()
    if current_price < flag_low * 0.95:
        return 0, {'pattern_broken': True, 'break_reason': 'Below flag support'}
    
    if current_price < start_price:
        return 0, {'pattern_broken': True, 'break_reason': 'Below flagpole start'}
    
    # Recency
    flag_high = flag_data['High'].max()
    days_old = next((i for i in range(1, 11) if data['High'].iloc[-i] == flag_high), 11)
    
    if days_old > 10:
        return confidence * 0.5, {**pattern_info, 'pattern_stale': True, 'days_old': days_old}
    
    pattern_info['days_since_high'] = days_old
    
    # Technical confirmation
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 15
        pattern_info['macd_bullish'] = True
    
    if histogram.iloc[-1] > histogram.iloc[-3]:
        confidence += 10
        pattern_info['momentum_recovering'] = True
    
    # Volume pattern
    flagpole_vol = data['Volume'].iloc[-flagpole_start:-flagpole_end].mean()
    flag_vol = flag_data['Volume'].mean()
    if flagpole_vol > flag_vol * 1.2:
        confidence += 15
        pattern_info['volume_pattern'] = True
    
    # Near breakout
    if current_price >= flag_high * 0.95:
        confidence += 10
        pattern_info['near_breakout'] = True
    
    return confidence, pattern_info

def detect_cup_handle(data, macd_line, signal_line, histogram):
    """Detect cup handle with strict handle requirements"""
    confidence = 0
    pattern_info = {}
    
    if len(data) < 40:
        return confidence, pattern_info
    
    # Strict handle sizing
    max_lookback = min(60, len(data) - 5)
    handle_days = min(10, max_lookback // 6)  # Max 10 days
    cup_days = max_lookback - handle_days
    
    cup_data = data.iloc[-max_lookback:-handle_days]
    handle_data = data.tail(handle_days)
    
    if len(cup_data) < 20:
        return confidence, pattern_info
    
    # Cup formation
    cup_start = cup_data['Close'].iloc[0]
    cup_bottom = cup_data['Low'].min()
    cup_right = cup_data['Close'].iloc[-1]
    cup_depth = (max(cup_start, cup_right) - cup_bottom) / max(cup_start, cup_right)
    
    if not (0.15 <= cup_depth <= 0.45 and cup_right >= cup_start * 0.90):
        return confidence, pattern_info
    
    confidence += 30
    pattern_info['cup_depth'] = f"{cup_depth*100:.1f}%"
    
    # Handle validation - VERY STRICT
    handle_low = handle_data['Low'].min()
    current_price = data['Close'].iloc[-1]
    handle_depth = (cup_right - handle_low) / cup_right
    
    # Handle too deep = pattern broken
    if handle_depth > 0.12:
        return 0, {'pattern_broken': True, 'break_reason': f'Handle too deep: {handle_depth*100:.1f}%'}
    
    if handle_depth <= 0.08:
        confidence += 25
        pattern_info['perfect_handle'] = f"{handle_depth*100:.1f}%"
    else:
        confidence += 15
        pattern_info['good_handle'] = f"{handle_depth*100:.1f}%"
    
    # Handle duration
    if handle_days > 8:
        return confidence * 0.6, {**pattern_info, 'handle_too_long': f"{handle_days} days"}
    
    if handle_days <= 5:
        confidence += 10
        pattern_info['short_handle'] = f"{handle_days} days"
    
    # Recency
    days_since_low = next((i for i in range(1, handle_days + 3) if data['Low'].iloc[-i] == handle_low), handle_days + 3)
    
    if days_since_low > handle_days + 2:
        return confidence * 0.5, {**pattern_info, 'pattern_stale': True, 'handle_age': days_since_low}
    
    # Pattern validation
    breakout_level = max(cup_start, cup_right)
    if current_price < breakout_level * 0.90:
        return confidence * 0.7, {**pattern_info, 'pattern_stale': True, 'far_from_rim': True}
    
    if current_price < handle_low * 0.97:
        return 0, {'pattern_broken': True, 'break_reason': 'Below handle support'}
    
    # Technical confirmation
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 10
        pattern_info['macd_bullish'] = True
    
    # Volume drying up
    if len(cup_data) > 15:
        cup_volume = cup_data['Volume'].mean()
        handle_volume = handle_data['Volume'].mean()
        if handle_volume < cup_volume * 0.80:
            confidence += 10
            pattern_info['volume_dryup'] = True
    
    return confidence, pattern_info

def detect_pattern(data, pattern_type):
    """Detect patterns with normalized scoring"""
    if len(data) < 30:
        return False, 0, {}
    
    # Add indicators
    data['RSI'] = calculate_rsi(data)
    macd_line, signal_line, histogram = calculate_macd(data)
    
    # Volume analysis
    data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
    
    confidence = 0
    pattern_info = {}
    
    if pattern_type == "Flat Top Breakout":
        confidence, pattern_info = detect_flat_top(data, macd_line, signal_line, histogram)
        confidence = min(confidence, 100)
        
    elif pattern_type == "Bull Flag":
        confidence, pattern_info = detect_bull_flag(data, macd_line, signal_line, histogram)
        confidence = min(confidence * 1.05, 100)
        
    elif pattern_type == "Cup Handle":
        confidence, pattern_info = detect_cup_handle(data, macd_line, signal_line, histogram)
        confidence = min(confidence * 1.1, 100)
    
    # Add technical data
    pattern_info['macd_line'] = macd_line
    pattern_info['signal_line'] = signal_line
    pattern_info['histogram'] = histogram
    
    return confidence >= 55, confidence, pattern_info

def calculate_levels(data, pattern_info, pattern_type):
    """Calculate entry, stop, targets"""
    current_price = data['Close'].iloc[-1]
    
    if pattern_type == "Flat Top Breakout":
        entry = pattern_info.get('resistance_level', current_price * 1.01)
        stop = data['Low'].tail(10).min() * 0.98
    elif pattern_type == "Bull Flag":
        entry = data['High'].tail(15).max() * 1.005
        stop = data['Low'].tail(10).min() * 0.98
    else:  # Cup Handle
        entry = current_price * 1.01
        stop = data.tail(10)['Low'].min() * 0.97
    
    target1 = entry + (entry - stop) * 2.0
    target2 = entry + (entry - stop) * 3.0
    
    return {
        'entry': entry,
        'stop': stop,
        'target1': target1,
        'target2': target2,
        'risk': entry - stop,
        'reward1': target1 - entry,
        'rr_ratio': (target1 - entry) / (entry - stop) if entry > stop else 0
    }

def create_chart(data, ticker, pattern_type, pattern_info, levels):
    """Create enhanced chart"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(f'{ticker} - {pattern_type}', 'MACD', 'Volume'),
        vertical_spacing=0.05,
        row_heights=[0.6, 0.25, 0.15]
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Moving averages
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SMA20'], name='SMA 20', 
                  line=dict(color='orange', width=1)),
        row=1, col=1
    )
    
    # Levels
    fig.add_hline(y=levels['entry'], line_color="green", 
                 annotation_text=f"Entry: ${levels['entry']:.2f}", row=1, col=1)
    fig.add_hline(y=levels['stop'], line_color="red", 
                 annotation_text=f"Stop: ${levels['stop']:.2f}", row=1, col=1)
    fig.add_hline(y=levels['target1'], line_color="lime", 
                 annotation_text=f"Target: ${levels['target1']:.2f}", row=1, col=1)
    
    # MACD
    macd_line = pattern_info['macd_line']
    signal_line = pattern_info['signal_line']
    histogram = pattern_info['histogram']
    
    fig.add_trace(go.Scatter(x=data.index, y=macd_line, name='MACD', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=signal_line, name='Signal', line=dict(color='red')), row=2, col=1)
    
    colors = ['green' if h >= 0 else 'red' for h in histogram]
    fig.add_trace(go.Bar(x=data.index, y=histogram, name='Histogram', marker_color=colors), row=2, col=1)
    fig.add_hline(y=0, line_color="black", row=2, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='blue', opacity=0.6), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    
    return fig

def main():
    st.title("🎯 Pro Pattern Detector v3.0")
    st.markdown("**Enhanced Pattern Recognition with MACD & Volume**")
    
    if not YFINANCE_AVAILABLE:
        st.warning("⚠️ **Demo Mode**: Using simulated data (yfinance not available)")
    
    st.error("""
    🚨 **DISCLAIMER**: Educational purposes only. Not financial advice. 
    Trading involves substantial risk. Consult professionals before trading.
    """)
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    patterns = ["Flat Top Breakout", "Bull Flag", "Cup Handle"]
    selected_patterns = st.sidebar.multiselect(
        "Select Patterns:", patterns, default=["Flat Top Breakout", "Bull Flag"]
    )
    
    tickers = st.sidebar.text_input("Tickers:", "AAPL,MSFT,NVDA")
    period = st.sidebar.selectbox("Period:", ["1mo", "3mo", "6mo", "1y"], index=1)
    min_confidence = st.sidebar.slider("Min Confidence:", 45, 85, 55)
    
    if st.sidebar.button("🚀 Analyze", type="primary"):
        if tickers and selected_patterns:
            ticker_list = [t.strip().upper() for t in tickers.split(',')]
            
            st.header("📈 Pattern Analysis Results")
            results = []
            
            for ticker in ticker_list:
                st.subheader(f"📊 {ticker}")
                
                data = get_stock_data(ticker, period)
                if data is not None and len(data) >= 50:
                    
                    # Detect all patterns first
                    all_patterns = {}
                    for pattern in selected_patterns:
                        detected, confidence, info = detect_pattern(data, pattern)
                        if detected and confidence >= min_confidence:
                            all_patterns[pattern] = {'confidence': confidence, 'info': info}
                    
                    # Show pattern conflicts
                    if len(all_patterns) > 1:
                        st.warning(f"⚠️ **Multiple patterns detected** - consider which is most dominant:")
                        for pat, details in all_patterns.items():
                            st.write(f"  • {pat}: {details['confidence']:.0f}%")
                    
                    # Display each pattern
                    for pattern in selected_patterns:
                        detected, confidence, info = detect_pattern(data, pattern)
                        
                        if detected and confidence >= min_confidence:
                            levels = calculate_levels(data, info, pattern)
                            
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                if confidence >= 80:
                                    st.success(f"✅ {pattern} DETECTED")
                                elif confidence >= 70:
                                    st.success(f"🟢 {pattern} DETECTED")
                                else:
                                    st.info(f"🟡 {pattern} DETECTED")
                                    
                                st.metric("Confidence", f"{confidence:.0f}%")
                                st.write(f"**Entry**: ${levels['entry']:.2f}")
                                st.write(f"**Stop**: ${levels['stop']:.2f}")
                                st.write(f"**Target**: ${levels['target1']:.2f}")
                                st.write(f"**R/R**: {levels['rr_ratio']:.1f}:1")
                            
                            with col2:
                                # Flat Top info
                                if info.get('initial_ascension'):
                                    st.write(f"🚀 Initial rise: {info['initial_ascension']}")
                                if info.get('descending_highs'):
                                    st.write("📉 Lower highs phase")
                                if info.get('higher_lows'):
                                    st.write("📈 Higher lows (triangle)")
                                if info.get('resistance_touches'):
                                    st.write(f"🔴 Resistance: {info['resistance_touches']} touches")
                                
                                # Bull Flag info
                                if info.get('flagpole_gain'):
                                    st.write(f"🚀 Flagpole: {info['flagpole_gain']}")
                                if info.get('healthy_pullback'):
                                    st.write(f"📉 Flag pullback: {info.get('flag_pullback', '')}")
                                if info.get('days_since_high'):
                                    st.write(f"⏰ Flag age: {info['days_since_high']} days")
                                
                                # Cup Handle info
                                if info.get('cup_depth'):
                                    st.write(f"☕ Cup depth: {info['cup_depth']}")
                                if info.get('perfect_handle'):
                                    st.success(f"✨ Perfect handle: {info['perfect_handle']}")
                                elif info.get('good_handle'):
                                    st.write(f"👍 Good handle: {info['good_handle']}")
                                if info.get('short_handle'):
                                    st.write(f"⚡ Short handle: {info['short_handle']}")
                                if info.get('handle_too_long'):
                                    st.warning(f"⚠️ Handle aging: {info['handle_too_long']}")
                                
                                # Warnings
                                if info.get('pattern_stale'):
                                    if info.get('days_old'):
                                        st.warning(f"⚠️ Pattern aging: {info['days_old']} days")
                                    elif info.get('handle_age'):
                                        st.warning(f"⚠️ Handle old: {info['handle_age']} days")
                                    elif info.get('far_from_rim'):
                                        st.warning("⚠️ Far from cup rim")
                                
                                if info.get('pattern_broken'):
                                    st.error(f"🚨 BROKEN: {info.get('break_reason', '')}")
                                
                                # Technical indicators
                                if info.get('macd_bullish'):
                                    st.write("📈 MACD bullish")
                                if info.get('momentum_recovering'):
                                    st.write("📈 Momentum recovering")
                                if info.get('volume_pattern'):
                                    st.write("📊 Volume pattern confirmed")
                                if info.get('volume_dryup'):
                                    st.write("💧 Volume drying up")
                                if info.get('near_breakout'):
                                    st.write("🎯 Near breakout")
                            
                            # Chart
                            fig = create_chart(data, ticker, pattern, info, levels)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add to results
                            results.append({
                                'Ticker': ticker,
                                'Pattern': pattern,
                                'Confidence': f"{confidence:.0f}%",
                                'Entry': f"${levels['entry']:.2f}",
                                'Stop': f"${levels['stop']:.2f}",
                                'Target': f"${levels['target1']:.2f}",
                                'R_R_Ratio': f"{levels['rr_ratio']:.1f}:1",
                                'Risk': f"${levels['risk']:.2f}"
                            })
                        else:
                            st.info(f"❌ {pattern}: {confidence:.0f}% (below threshold)")
                else:
                    st.error(f"❌ Insufficient data for {ticker}")
            
            # Summary
            if results:
                st.header("📋 Summary")
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Patterns", len(results))
                with col2:
                    scores = [int(r['Confidence'].replace('%', '')) for r in results]
                    avg_score = sum(scores) / len(scores) if scores else 0
                    st.metric("Avg Confidence", f"{avg_score:.0f}%")
                with col3:
                    ratios = [float(r['R_R_Ratio'].split(':')[0]) for r in results]
                    avg_rr = sum(ratios) / len(ratios) if ratios else 0
                    st.metric("Avg R/R", f"{avg_rr:.1f}:1")
                
                # Download
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    f"patterns_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
            else:
                st.info("🔍 No patterns detected. Try lowering the confidence threshold.")

if __name__ == "__main__":
    main()
