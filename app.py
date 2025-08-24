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

def detect_ascending_triangle(data, macd_line, signal_line, histogram):
    """Detect ascending triangle: Higher lows + Flat resistance (continuation pattern)"""
    confidence = 0
    pattern_info = {}
    
    if len(data) < 30:
        return confidence, pattern_info
    
    # STEP 1: Identify flat resistance level (similar to flat top)
    # Look for recent highs that form flat resistance
    recent_period = min(40, len(data))
    resistance_data = data.tail(recent_period)
    
    # Find the high point that's being tested multiple times
    resistance_high = resistance_data['High'].max()
    
    # Look for multiple touches of this level (within 2%)
    resistance_touches = sum(1 for h in resistance_data['High'].tail(20) if h >= resistance_high * 0.98)
    
    if resistance_touches < 2:  # Need at least 2 touches
        return confidence, pattern_info
    
    confidence += 25
    pattern_info['resistance_level'] = resistance_high
    pattern_info['resistance_touches'] = resistance_touches
    
    # STEP 2: Critical - Higher lows pattern (ascending support)
    # This is what makes it ascending triangle vs flat top
    support_data = data.tail(25)  # Last 25 days for support analysis
    support_lows = support_data['Low'].rolling(3, center=True).min().dropna()
    
    if len(support_lows) < 4:
        return confidence, pattern_info
    
    # Check for clear ascending pattern in lows
    first_low = support_lows.iloc[0]
    last_low = support_lows.iloc[-1]
    middle_low = support_lows.iloc[len(support_lows)//2]
    
    # Strong higher lows pattern
    if last_low > first_low * 1.02 and middle_low > first_low * 1.01:
        confidence += 30
        pattern_info['strong_higher_lows'] = True
        pattern_info['support_trend'] = 'strongly ascending'
    elif last_low > first_low * 1.005:  # Moderate higher lows
        confidence += 20
        pattern_info['moderate_higher_lows'] = True
        pattern_info['support_trend'] = 'ascending'
    else:
        # Not really ascending - reduce confidence
        return confidence * 0.5, pattern_info
    
    # STEP 3: Triangle compression (narrowing range)
    early_range = (resistance_data['High'].iloc[:10] - resistance_data['Low'].iloc[:10]).mean()
    recent_range = (resistance_data['High'].tail(10) - resistance_data['Low'].tail(10)).mean()
    
    if recent_range < early_range * 0.8:  # Range is compressing
        confidence += 15
        pattern_info['compression'] = True
        pattern_info['range_compression'] = f"{(1 - recent_range/early_range)*100:.1f}%"
    
    # STEP 4: Duration validation (good triangles take time)
    triangle_duration = len(resistance_data)
    if triangle_duration >= 15:  # At least 15 days
        confidence += 10
        pattern_info['sufficient_duration'] = f"{triangle_duration} days"
    
    # STEP 5: Breakout proximity (should be near resistance)
    current_price = data['Close'].iloc[-1]
    distance_to_resistance = (resistance_high - current_price) / current_price
    
    if distance_to_resistance <= 0.03:  # Within 3% of resistance
        confidence += 15
        pattern_info['near_breakout'] = True
        pattern_info['breakout_proximity'] = f"{distance_to_resistance*100:.1f}%"
    elif distance_to_resistance <= 0.05:  # Within 5%
        confidence += 10
        pattern_info['approaching_breakout'] = True
    
    # STEP 6: Pattern recency validation
    days_since_resistance_touch = next(
        (i for i in range(1, 11) if data['High'].iloc[-i] >= resistance_high * 0.98), 11
    )
    
    if days_since_resistance_touch > 8:  # Too stale
        confidence *= 0.7
        pattern_info['pattern_stale'] = True
        pattern_info['days_old'] = days_since_resistance_touch
    
    # STEP 7: Support break invalidation
    recent_support = support_lows.iloc[-3:].min()
    if current_price < recent_support * 0.97:
        return 0, {'pattern_broken': True, 'break_reason': 'Below ascending support'}
    
    # STEP 8: MACD confirmation (momentum building)
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 10
        pattern_info['macd_bullish'] = True
    
    if histogram.iloc[-1] > histogram.iloc[-2]:
        confidence += 10
        pattern_info['momentum_building'] = True
    
    # STEP 9: Volume analysis (should be decreasing then surge on breakout)
    if len(data) > 20:
        early_volume = data['Volume'].iloc[-25:-15].mean() if len(data) >= 25 else data['Volume'].iloc[:-10].mean()
        recent_volume = data['Volume'].tail(10).mean()
        
        if recent_volume < early_volume * 0.85:  # Volume drying up
            confidence += 10
            pattern_info['volume_compression'] = True
        
        # Check for recent volume surge (potential breakout)
        if data['Volume'].tail(3).mean() > recent_volume * 1.3:
            confidence += 8
            pattern_info['volume_surge'] = True
    
    return confidence, pattern_info
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
    """Detect cup handle with much more lenient requirements"""
    confidence = 0
    pattern_info = {}
    
    if len(data) < 30:  # Reduced from 40
        return confidence, pattern_info
    
    # STEP 1: Much more flexible sizing
    max_lookback = min(100, len(data) - 3)  # Increased from 80
    
    # Handle can be much longer - up to 30 days
    handle_days = min(30, max_lookback // 3)  # Even more flexible
    cup_days = max_lookback - handle_days
    
    cup_data = data.iloc[-max_lookback:-handle_days] if handle_days > 0 else data.iloc[-max_lookback:]
    handle_data = data.tail(handle_days) if handle_days > 0 else data.tail(5)
    
    if len(cup_data) < 15:  # Very minimal requirement
        return confidence, pattern_info
    
    # STEP 2: Very lenient cup formation
    cup_start = cup_data['Close'].iloc[0]
    cup_bottom = cup_data['Low'].min()
    cup_right = cup_data['Close'].iloc[-1]
    cup_depth = (max(cup_start, cup_right) - cup_bottom) / max(cup_start, cup_right)
    
    # Much more lenient cup requirements
    if cup_depth < 0.08 or cup_depth > 0.60:  # Very wide range
        return confidence, pattern_info
    
    if cup_right < cup_start * 0.75:  # Very lenient rim requirement
        return confidence, pattern_info
    
    confidence += 25  # Base points for having a cup
    pattern_info['cup_depth'] = f"{cup_depth*100:.1f}%"
    
    # STEP 3: Very lenient handle validation
    if handle_days > 0:
        handle_low = handle_data['Low'].min()
        current_price = data['Close'].iloc[-1]
        handle_depth = (cup_right - handle_low) / cup_right
        
        # Much more lenient handle depth - up to 25%!
        if handle_depth > 0.25:  # Increased from 0.18
            confidence += 10  # Small penalty instead of rejection
            pattern_info['deep_handle'] = f"{handle_depth*100:.1f}%"
        elif handle_depth <= 0.08:  # Perfect handle
            confidence += 20
            pattern_info['perfect_handle'] = f"{handle_depth*100:.1f}%"
        elif handle_depth <= 0.15:  # Good handle
            confidence += 15
            pattern_info['good_handle'] = f"{handle_depth*100:.1f}%"
        else:  # Acceptable handle (15-25%)
            confidence += 10
            pattern_info['acceptable_handle'] = f"{handle_depth*100:.1f}%"
        
        # Handle duration - very lenient
        if handle_days > 25:  # Much more lenient
            confidence *= 0.8  # Small penalty instead of major one
            pattern_info['long_handle'] = f"{handle_days} days"
        elif handle_days <= 10:  # Short handle bonus
            confidence += 10
            pattern_info['short_handle'] = f"{handle_days} days"
        elif handle_days <= 20:  # Medium handle
            confidence += 5
            pattern_info['medium_handle'] = f"{handle_days} days"
        
    else:
        # No clear handle, but still might be forming
        confidence += 10
        pattern_info['forming_handle'] = "Handle forming"
    
    # STEP 4: Very lenient recency
    current_price = data['Close'].iloc[-1]
    
    # STEP 5: Very lenient pattern validation
    breakout_level = max(cup_start, cup_right)
    if current_price < breakout_level * 0.70:  # Very lenient
        confidence *= 0.7
        pattern_info['far_from_rim'] = True
    else:
        confidence += 5  # Bonus for being near rim
    
    # STEP 6: No strict breakage rules - just reduce confidence
    if handle_days > 0:
        handle_low = handle_data['Low'].min()
        if current_price < handle_low * 0.90:  # Very lenient
            confidence *= 0.8  # Reduce but don't eliminate
            pattern_info['below_handle'] = True
    
    # STEP 7: Technical confirmation (bonus points)
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 10
        pattern_info['macd_bullish'] = True
    
    # Volume analysis (bonus points)
    if len(cup_data) > 10 and handle_days > 0:
        cup_volume = cup_data['Volume'].mean()
        handle_volume = handle_data['Volume'].mean()
        if handle_volume < cup_volume * 0.85:
            confidence += 8
            pattern_info['volume_dryup'] = True
    
    # STEP 8: Minimum confidence check
    if confidence < 35:  # Very low minimum
        return confidence, pattern_info
    
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
        
    elif pattern_type == "Ascending Triangle":
        confidence, pattern_info = detect_ascending_triangle(data, macd_line, signal_line, histogram)
        confidence = min(confidence * 1.05, 100)  # Small boost for competitiveness
        
    elif pattern_type == "Ascending Triangle":
        # Entry at resistance breakout
        entry = pattern_info.get('resistance_level', current_price * 1.01)
        
        # Stop below recent ascending support
        recent_support_lows = data['Low'].tail(15).rolling(3, center=True).min().dropna()
        if len(recent_support_lows) > 0:
            support_level = recent_support_lows.iloc[-1]  # Latest support level
        else:
            support_level = data['Low'].tail(15).min()
        
        volatility_stop = entry - volatility_stop_distance
        traditional_stop = support_level * 0.98
        
        # Use the higher stop (better R/R)
        stop = max(volatility_stop, traditional_stop)
        
        # Ensure proper stop distance
        min_stop_distance = entry * 0.04  # At least 4% for triangles
        if stop >= entry:
            stop = entry - min_stop_distance
        elif (entry - stop) < min_stop_distance:
            stop = entry - min_stop_distance
        
        # MEASURED MOVE: Triangle height projection
        triangle_height = entry - support_level
        
        # Ensure minimum triangle height
        triangle_height = max(triangle_height, entry * 0.06)  # At least 6%
        
        target1 = entry + triangle_height  # 1:1 measured move
        target2 = entry + (triangle_height * 1.618)  # Golden ratio extension
        
        target_method = "Ascending Triangle Height Projection"
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
    """Calculate entry, stop, targets using MEASURED MOVES with improved R/R ratios"""
    current_price = data['Close'].iloc[-1]
    
    # Calculate a reasonable stop distance based on recent volatility
    recent_range = data['High'].tail(20) - data['Low'].tail(20)
    avg_range = recent_range.mean()
    volatility_stop_distance = avg_range * 1.5  # 1.5x average daily range
    
    if pattern_type == "Flat Top Breakout":
        # Entry at resistance breakout
        entry = pattern_info.get('resistance_level', current_price * 1.01)
        
        # Improved stop calculation
        recent_low = data['Low'].tail(15).min()
        volatility_stop = entry - volatility_stop_distance
        traditional_stop = recent_low * 0.98
        
        # Use the higher of the two (closer stop for better R/R)
        stop = max(volatility_stop, traditional_stop)
        
        # Ensure stop is below entry with minimum distance
        min_stop_distance = entry * 0.03  # At least 3% below entry
        if stop >= entry:
            stop = entry - min_stop_distance
        elif (entry - stop) < min_stop_distance:
            stop = entry - min_stop_distance
        
        # MEASURED MOVE: Triangle height projection
        if 'resistance_level' in pattern_info:
            # Calculate triangle height more accurately
            support_level = data['Low'].tail(20).max()  # Recent support
            triangle_height = entry - support_level
            
            # Ensure minimum triangle height
            triangle_height = max(triangle_height, entry * 0.05)  # At least 5%
            
            target1 = entry + triangle_height  # 1:1 measured move
            target2 = entry + (triangle_height * 1.618)  # Golden ratio extension
        else:
            # Fallback with guaranteed good R/R
            risk = entry - stop
            target1 = entry + (risk * 2.0)
            target2 = entry + (risk * 3.5)
        
        target_method = "Triangle Height Projection"
        
    elif pattern_type == "Bull Flag":
        # Entry at flag breakout
        flag_high = data['High'].tail(15).max()
        entry = flag_high * 1.005
        
        # Improved stop for bull flags
        flag_low = data['Low'].tail(12).min()  # Recent flag support
        volatility_stop = entry - volatility_stop_distance
        traditional_stop = flag_low * 0.98
        
        # Use the higher stop (better R/R)
        stop = max(volatility_stop, traditional_stop)
        
        # Ensure proper stop distance
        min_stop_distance = entry * 0.04  # At least 4% for bull flags
        if stop >= entry:
            stop = entry - min_stop_distance
        elif (entry - stop) < min_stop_distance:
            stop = entry - min_stop_distance
        
        # MEASURED MOVE: Enhanced flagpole calculation
        if 'flagpole_gain' in pattern_info:
            try:
                # Extract flagpole percentage more reliably
                flagpole_pct_str = pattern_info['flagpole_gain'].replace('%', '')
                flagpole_pct = float(flagpole_pct_str) / 100
                
                # Calculate flagpole height in dollars
                flagpole_start_price = entry / (1 + flagpole_pct)  # Estimate start price
                flagpole_height = entry - flagpole_start_price
                
                # Ensure minimum flagpole height
                flagpole_height = max(flagpole_height, entry * 0.08)  # At least 8%
                
                target1 = entry + flagpole_height  # Full flagpole projection
                target2 = entry + (flagpole_height * 1.382)  # Fibonacci extension
            except (ValueError, KeyError):
                # Fallback calculation
                risk = entry - stop
                target1 = entry + (risk * 2.5)
                target2 = entry + (risk * 4.0)
        else:
            # Fallback with good R/R
            risk = entry - stop
            target1 = entry + (risk * 2.5)
            target2 = entry + (risk * 4.0)
        
        target_method = "Flagpole Height Projection"
        
    elif pattern_type == "Cup Handle":
        # Improved cup handle entry calculation
        if 'cup_depth' in pattern_info:
            try:
                # Extract cup depth percentage
                cup_depth_str = pattern_info['cup_depth'].replace('%', '')
                cup_depth_pct = float(cup_depth_str) / 100
                
                # Estimate cup rim level more accurately
                # If current price dropped from rim, estimate rim level
                estimated_rim = current_price / (1 - cup_depth_pct * 0.3)  # Conservative estimate
                entry = estimated_rim * 1.005
            except (ValueError, KeyError):
                entry = current_price * 1.02  # Fallback
        else:
            entry = current_price * 1.02
        
        # Improved stop for cup handles
        handle_low = data.tail(15)['Low'].min()
        volatility_stop = entry - volatility_stop_distance
        traditional_stop = handle_low * 0.97
        
        # Use higher stop for better R/R
        stop = max(volatility_stop, traditional_stop)
        
        # Ensure proper stop distance
        min_stop_distance = entry * 0.05  # At least 5% for cup handles
        if stop >= entry:
            stop = entry - min_stop_distance
        elif (entry - stop) < min_stop_distance:
            stop = entry - min_stop_distance
        
        # MEASURED MOVE: Improved cup depth projection
        if 'cup_depth' in pattern_info:
            try:
                cup_depth_str = pattern_info['cup_depth'].replace('%', '')
                cup_depth_pct = float(cup_depth_str) / 100
                
                # Cup depth in dollars
                cup_depth_dollars = entry * cup_depth_pct
                
                # Ensure minimum measured move
                cup_depth_dollars = max(cup_depth_dollars, entry * 0.10)  # At least 10%
                
                target1 = entry + cup_depth_dollars  # Full cup depth projection
                target2 = entry + (cup_depth_dollars * 1.618)  # Golden ratio
            except (ValueError, KeyError):
                # Fallback
                risk = entry - stop
                target1 = entry + (risk * 2.0)
                target2 = entry + (risk * 3.0)
        else:
            # Fallback with good R/R
            risk = entry - stop
            target1 = entry + (risk * 2.0)
            target2 = entry + (risk * 3.0)
        
        target_method = "Cup Depth Projection"
    
    else:
        # Fallback for any other patterns
        entry = current_price * 1.01
        stop = current_price * 0.95  # 5% stop
        target1 = entry + (entry - stop) * 2.0
        target2 = entry + (entry - stop) * 3.0
        target_method = "Traditional 2:1 & 3:1"
    
    # Final safety checks to ensure good R/R ratios
    risk_amount = entry - stop
    reward1 = target1 - entry
    reward2 = target2 - entry
    
    # If R/R is too low, adjust targets upward
    if risk_amount > 0:
        rr1 = reward1 / risk_amount
        rr2 = reward2 / risk_amount
        
        # Ensure minimum 1.5:1 R/R for target 1
        if rr1 < 1.5:
            target1 = entry + (risk_amount * 1.5)
            reward1 = target1 - entry
            rr1 = 1.5
        
        # Ensure minimum 2.5:1 R/R for target 2
        if rr2 < 2.5:
            target2 = entry + (risk_amount * 2.5)
            reward2 = target2 - entry
            rr2 = 2.5
    else:
        # Emergency fallback if risk calculation fails
        risk_amount = entry * 0.05  # 5% risk
        stop = entry - risk_amount
        target1 = entry + (risk_amount * 2.0)
        target2 = entry + (risk_amount * 3.0)
        reward1 = target1 - entry
        reward2 = target2 - entry
        rr1 = 2.0
        rr2 = 3.0
    
    return {
        'entry': entry,
        'stop': stop,
        'target1': target1,
        'target2': target2,
        'risk': risk_amount,
        'reward1': reward1,
        'reward2': reward2,
        'rr_ratio1': reward1 / risk_amount if risk_amount > 0 else 0,
        'rr_ratio2': reward2 / risk_amount if risk_amount > 0 else 0,
        'target_method': target_method,
        'measured_move': True,
        'volatility_adjusted': True  # New flag
    }

def create_chart(data, ticker, pattern_type, pattern_info, levels):
    """Create enhanced chart with measured move annotations"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f'{ticker} - {pattern_type} | {levels["target_method"]}',
            'MACD Analysis', 
            'Volume Profile'
        ),
        vertical_spacing=0.05,
        row_heights=[0.6, 0.25, 0.15]
    )
    
    # Candlestick chart
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
    
    # Trading levels with enhanced annotations
    fig.add_hline(y=levels['entry'], line_color="green", line_width=2,
                 annotation_text=f"📈 Entry: ${levels['entry']:.2f}", row=1, col=1)
    fig.add_hline(y=levels['stop'], line_color="red", line_width=2,
                 annotation_text=f"🛑 Stop: ${levels['stop']:.2f}", row=1, col=1)
    fig.add_hline(y=levels['target1'], line_color="lime", line_width=2,
                 annotation_text=f"🎯 Target 1: ${levels['target1']:.2f} ({levels['rr_ratio1']:.1f}:1)", row=1, col=1)
    fig.add_hline(y=levels['target2'], line_color="darkgreen", line_width=1,
                 annotation_text=f"🎯 Target 2: ${levels['target2']:.2f} ({levels['rr_ratio2']:.1f}:1)", row=1, col=1)
    
    # Add pattern-specific annotations
    if pattern_type == "Bull Flag" and 'flagpole_gain' in pattern_info:
        flagpole_height = levels['reward1']  # This is the measured move
        fig.add_annotation(
            x=data.index[-5], y=levels['target1'],
            text=f"Measured Move: ${flagpole_height:.2f}",
            showarrow=True, arrowhead=2, arrowcolor="lime",
            bgcolor="rgba(0,255,0,0.1)", bordercolor="lime"
        )
    
    elif pattern_type == "Cup Handle" and 'cup_depth' in pattern_info:
        cup_move = levels['reward1']
        fig.add_annotation(
            x=data.index[-5], y=levels['target1'],
            text=f"Cup Depth Move: ${cup_move:.2f}",
            showarrow=True, arrowhead=2, arrowcolor="lime",
            bgcolor="rgba(0,255,0,0.1)", bordercolor="lime"
        )
    
    elif pattern_type == "Flat Top Breakout":
        triangle_height = levels['reward1']
        fig.add_annotation(
            x=data.index[-5], y=levels['target1'],
            text=f"Triangle Height: ${triangle_height:.2f}",
            showarrow=True, arrowhead=2, arrowcolor="lime",
            bgcolor="rgba(0,255,0,0.1)", bordercolor="lime"
        )
    
    # MACD chart
    macd_line = pattern_info['macd_line']
    signal_line = pattern_info['signal_line']
    histogram = pattern_info['histogram']
    
    fig.add_trace(go.Scatter(x=data.index, y=macd_line, name='MACD', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=signal_line, name='Signal', line=dict(color='red')), row=2, col=1)
    
    colors = ['green' if h >= 0 else 'red' for h in histogram]
    fig.add_trace(go.Bar(x=data.index, y=histogram, name='Histogram', marker_color=colors, opacity=0.6), row=2, col=1)
    fig.add_hline(y=0, line_color="black", row=2, col=1)
    
    # Volume chart
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='blue', opacity=0.6), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    
    return fig

def main():
    st.title("🎯 Pro Pattern Detector v4.0")
    st.markdown("**Measured Move Targets** - Professional Pattern Recognition with Geometric Projections")
    
    if not YFINANCE_AVAILABLE:
        st.warning("⚠️ **Demo Mode**: Using simulated data (yfinance not available)")
    
    st.error("""
    🚨 **DISCLAIMER**: Educational purposes only. Not financial advice. 
    Trading involves substantial risk. Consult professionals before trading.
    """)
    
    # Info box about new features
    with st.expander("🆕 What's New in v4.0"):
        st.markdown("""
        ### 🎯 **Professional Measured Move Targets**
        
        Instead of fixed 2:1 and 3:1 ratios, v4.0 uses **geometric projections**:
        
        **🚀 Bull Flag**: Target = Entry + Flagpole Height
        - *If flagpole is $10, target is $10 above breakout*
        
        **☕ Cup-Handle**: Target = Cup Rim + Cup Depth  
        - *If cup is 20% deep, target is 20% above rim*
        
        **📐 Flat Top**: Target = Breakout + Triangle Height
        - *Projects the triangle's height above resistance*
        
        **🔺 Ascending Triangle**: Target = Breakout + Triangle Height *(NEW!)*
        - *Continuation pattern with flat resistance + higher lows*
        
        ### 📊 **Enhanced Features**
        - **Volatility-adjusted stops** for better R/R ratios
        - **Guaranteed minimum 1.5:1** risk/reward ratios  
        - **Dual targets** with Fibonacci extensions
        - **Pattern invalidation** detection
        - **4 professional patterns** with measured moves
        
        **This is how professional traders set their targets!** 🎖️
        """)
    
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    patterns = ["Flat Top Breakout", "Ascending Triangle", "Bull Flag", "Cup Handle"]
    selected_patterns = st.sidebar.multiselect(
        "Select Patterns:", patterns, default=["Flat Top Breakout", "Ascending Triangle", "Bull Flag"]
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
                                
                                # Enhanced display with measured moves
                                st.write("**📊 Trading Levels:**")
                                st.write(f"**Entry**: ${levels['entry']:.2f}")
                                st.write(f"**Stop**: ${levels['stop']:.2f}")
                                st.write(f"**Target 1**: ${levels['target1']:.2f}")
                                st.write(f"**Target 2**: ${levels['target2']:.2f}")
                                
                                # NEW: Show R/R ratios for both targets
                                st.write("**🎯 Risk/Reward:**")
                                st.write(f"**T1 R/R**: {levels['rr_ratio1']:.1f}:1")
                                st.write(f"**T2 R/R**: {levels['rr_ratio2']:.1f}:1")
                                
                                # NEW: Show calculation method
                                st.info(f"📐 **Method**: {levels['target_method']}")
                            
                            with col2:
                                # Enhanced pattern-specific information with R/R validation
                                if info.get('initial_ascension'):
                                    st.write(f"🚀 Initial rise: {info['initial_ascension']}")
                                if info.get('descending_highs'):
                                    st.write("📉 Lower highs phase")
                                if info.get('higher_lows'):
                                    st.write("📈 Higher lows (triangle)")
                                if info.get('resistance_touches'):
                                    st.write(f"🔴 Resistance: {info['resistance_touches']} touches")
                                
                                # Ascending Triangle specific info
                                if info.get('strong_higher_lows'):
                                    st.success("📈 Strong ascending support")
                                elif info.get('moderate_higher_lows'):
                                    st.write("📈 Moderate ascending support")
                                if info.get('compression'):
                                    compression_pct = info.get('range_compression', '')
                                    st.write(f"🔥 Compression: {compression_pct}")
                                if info.get('sufficient_duration'):
                                    st.write(f"⏰ Duration: {info['sufficient_duration']}")
                                if info.get('volume_compression'):
                                    st.write("💧 Volume drying up")
                                if info.get('volume_surge'):
                                    st.write("🔥 Volume surge detected")
                                if info.get('near_breakout'):
                                    proximity = info.get('breakout_proximity', '')
                                    st.write(f"🎯 Near breakout: {proximity}")
                                elif info.get('approaching_breakout'):
                                    st.write("🎯 Approaching breakout")
                                
                                # Show measured move for Ascending Triangle
                                if pattern == "Ascending Triangle" and levels.get('measured_move'):
                                    triangle_height = levels['reward1']
                                    st.success(f"📐 **Triangle Height**: ${triangle_height:.2f}")
                                    st.write("*Target = Breakout + Triangle Height*")
                                    
                                    # R/R validation warning
                                    if levels['rr_ratio1'] < 1.5:
                                        st.warning(f"⚠️ Low R/R: {levels['rr_ratio1']:.1f}:1 (adjusted to 1.5:1)")
                                
                                # Bull Flag with measured move explanation
                                if info.get('flagpole_gain'):
                                    flagpole_pct = info['flagpole_gain']
                                    flagpole_dollars = levels['reward1']
                                    st.write(f"🚀 Flagpole: {flagpole_pct}")
                                    st.success(f"📐 **Measured Move**: ${flagpole_dollars:.2f}")
                                    st.write("*Target = Entry + Flagpole Height*")
                                    
                                    # R/R validation warning
                                    if levels['rr_ratio1'] < 1.5:
                                        st.warning(f"⚠️ Low R/R: {levels['rr_ratio1']:.1f}:1 (adjusted to 1.5:1)")
                                
                                if info.get('healthy_pullback'):
                                    st.write(f"📉 Flag pullback: {info.get('flag_pullback', '')}")
                                if info.get('days_since_high'):
                                    st.write(f"⏰ Flag age: {info['days_since_high']} days")
                                
                                # Cup Handle with measured move explanation
                                if info.get('cup_depth'):
                                    cup_pct = info['cup_depth']
                                    cup_dollars = levels['reward1']
                                    st.write(f"☕ Cup depth: {cup_pct}")
                                    st.success(f"📐 **Measured Move**: ${cup_dollars:.2f}")
                                    st.write("*Target = Rim + Cup Depth*")
                                    
                                    # R/R validation warning
                                    if levels['rr_ratio1'] < 1.5:
                                        st.warning(f"⚠️ Low R/R: {levels['rr_ratio1']:.1f}:1 (adjusted to 1.5:1)")
                                
                                if info.get('perfect_handle'):
                                    st.success(f"✨ Perfect handle: {info['perfect_handle']}")
                                elif info.get('good_handle'):
                                    st.write(f"👍 Good handle: {info['good_handle']}")
                                elif info.get('acceptable_handle'):
                                    st.write(f"✅ Acceptable handle: {info['acceptable_handle']}")
                                elif info.get('deep_handle'):
                                    st.warning(f"⚠️ Deep handle: {info['deep_handle']}")
                                
                                if info.get('short_handle'):
                                    st.write(f"⚡ Short handle: {info['short_handle']}")
                                elif info.get('medium_handle'):
                                    st.write(f"⏰ Medium handle: {info['medium_handle']}")
                                elif info.get('long_handle'):
                                    st.warning(f"⚠️ Long handle: {info['long_handle']}")
                                
                                # Flat Top with triangle height explanation
                                if pattern == "Flat Top Breakout" and levels.get('measured_move'):
                                    triangle_height = levels['reward1']
                                    st.success(f"📐 **Triangle Height**: ${triangle_height:.2f}")
                                    st.write("*Target = Breakout + Triangle Height*")
                                    
                                    # R/R validation warning
                                    if levels['rr_ratio1'] < 1.5:
                                        st.warning(f"⚠️ Low R/R: {levels['rr_ratio1']:.1f}:1 (adjusted to 1.5:1)")
                                
                                # Show if volatility-adjusted stops were used
                                if levels.get('volatility_adjusted'):
                                    st.info("📊 Volatility-adjusted stops")
                                
                                # Warnings and status
                                if info.get('handle_too_long'):
                                    st.warning(f"⚠️ Handle aging: {info['handle_too_long']}")
                                
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
                            
                            # Add to results with enhanced information
                            results.append({
                                'Ticker': ticker,
                                'Pattern': pattern,
                                'Confidence': f"{confidence:.0f}%",
                                'Entry': f"${levels['entry']:.2f}",
                                'Stop': f"${levels['stop']:.2f}",
                                'Target 1': f"${levels['target1']:.2f}",
                                'Target 2': f"${levels['target2']:.2f}",
                                'R/R 1': f"{levels['rr_ratio1']:.1f}:1",
                                'R/R 2': f"{levels['rr_ratio2']:.1f}:1",
                                'Risk': f"${levels['risk']:.2f}",
                                'Method': levels['target_method']
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
                    if results:
                        ratios = [float(r['R/R 1'].split(':')[0]) for r in results]
                        avg_rr = sum(ratios) / len(ratios) if ratios else 0
                        st.metric("Avg R/R T1", f"{avg_rr:.1f}:1")
                
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
