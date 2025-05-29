#!/usr/bin/env python3
"""Debug script to check indicator values at the first test bar."""

import json
import pandas as pd
from datetime import datetime

# Load the data
data = pd.read_csv("data/SPY_1min.csv")
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Get test data (bars 800-999)
test_data = data.iloc[800:1000].copy()
first_bar = test_data.iloc[0]

print(f"First test bar: {first_bar['timestamp']} (index {test_data.index[0]})")
print(f"OHLCV: O={first_bar['open']:.2f}, H={first_bar['high']:.2f}, L={first_bar['low']:.2f}, C={first_bar['close']:.2f}, V={first_bar['volume']}")

# Calculate indicators at first bar using training data for warmup
train_data = data.iloc[0:800].copy()

# MA Trend (20-period SMA slope)
ma_period = 20
if len(train_data) >= ma_period:
    # Get last 20 closes from training data
    last_closes = train_data['close'].iloc[-ma_period:].values
    ma_current = last_closes.mean()
    
    # Get previous MA (drop first, add first test bar)
    prev_closes = train_data['close'].iloc[-(ma_period-1):].values
    ma_prev = prev_closes.mean()
    
    ma_trend = (ma_current - ma_prev) / ma_prev if ma_prev > 0 else 0
    print(f"\nMA Trend calculation:")
    print(f"  Current MA (last 20 from train): {ma_current:.4f}")
    print(f"  Previous MA: {ma_prev:.4f}")
    print(f"  MA Trend: {ma_trend:.6f}")
else:
    print(f"\nNot enough training data for MA calculation")

# Volume Trend (10-period average)
vol_period = 10
if len(train_data) >= vol_period:
    avg_volume = train_data['volume'].iloc[-vol_period:].mean()
    current_volume = first_bar['volume']
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
    print(f"\nVolume calculation:")
    print(f"  Current volume: {current_volume}")
    print(f"  Average volume (last 10): {avg_volume:.2f}")
    print(f"  Volume ratio: {volume_ratio:.4f}")

# Check what trending_up threshold would be
print(f"\nRegime detection thresholds:")
print(f"  Trending up: ma_trend > 0.0001 AND volume_ratio > 0.8")
print(f"  Trending down: ma_trend < -0.0001 AND volume_ratio > 0.8")
print(f"  Default: otherwise")

# Determine regime
if ma_trend > 0.0001 and volume_ratio > 0.8:
    regime = "trending_up"
elif ma_trend < -0.0001 and volume_ratio > 0.8:
    regime = "trending_down"
else:
    regime = "default"

print(f"\nExpected regime: {regime}")

# Also check RSI (14-period)
rsi_period = 14
if len(train_data) >= rsi_period + 1:
    closes = train_data['close'].iloc[-(rsi_period+1):].values
    deltas = closes[1:] - closes[:-1]
    gains = deltas.copy()
    losses = deltas.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = -losses
    
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    
    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    print(f"\nRSI calculation:")
    print(f"  RSI value: {rsi:.2f}")
    print(f"  Overbought: > 70")
    print(f"  Oversold: < 30")