#!/usr/bin/env python3
"""
Test solution for matching optimizer signals by handling warmup properly.

Key insight: We need to suppress signal generation during warmup phase
and only generate signals during test phase, but with warmed indicators.
"""

import pandas as pd
import numpy as np
from collections import deque

def simulate_indicator_warmup():
    """
    Simulate how indicators warm up to understand the exact state needed.
    """
    print("INDICATOR WARMUP SIMULATION")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('data/1000_1min.csv')
    
    # Split at 80%
    split_idx = int(len(df) * 0.8)
    train_prices = df['Close'].values[:split_idx]
    test_prices = df['Close'].values[split_idx:]
    
    print(f"Total bars: {len(df)}")
    print(f"Training bars: {len(train_prices)} (0 to {split_idx-1})")
    print(f"Test bars: {len(test_prices)} ({split_idx} to {len(df)-1})")
    
    # Simulate MA indicator warmup
    short_window = 10
    long_window = 20
    
    # Warm up on training data
    short_buffer = deque(maxlen=short_window)
    long_buffer = deque(maxlen=long_window)
    
    # Process training data
    for i, price in enumerate(train_prices):
        short_buffer.append(price)
        long_buffer.append(price)
    
    # State at end of training
    print(f"\nIndicator state after training:")
    print(f"Short buffer ({short_window} bars): {list(short_buffer)}")
    print(f"Long buffer ({long_window} bars): {list(long_buffer)}")
    print(f"MA short: {np.mean(short_buffer):.4f}")
    print(f"MA long: {np.mean(long_buffer):.4f}")
    
    # Process first test bar
    first_test_price = test_prices[0]
    short_buffer.append(first_test_price)
    long_buffer.append(first_test_price)
    
    ma_short = np.mean(short_buffer)
    ma_long = np.mean(long_buffer)
    
    print(f"\nFirst test bar:")
    print(f"Price: {first_test_price:.2f}")
    print(f"MA short: {ma_short:.4f}")
    print(f"MA long: {ma_long:.4f}")
    
    # Check signal condition
    if first_test_price > ma_short > ma_long:
        print("Signal: BUY (price > MA_short > MA_long)")
    elif first_test_price < ma_short < ma_long:
        print("Signal: SELL (price < MA_short < MA_long)")
    else:
        print("Signal: NONE")
    
    # Show test timestamps
    test_timestamps = pd.to_datetime(df['timestamp'].values[split_idx:split_idx+5])
    print(f"\nFirst 5 test timestamps:")
    for i, ts in enumerate(test_timestamps):
        print(f"  {i}: {ts}")

def create_warmup_aware_strategy():
    """
    Create a strategy configuration that's aware of warmup needs.
    """
    strategy_patch = '''
# Strategy modification to handle warmup phase
# Add this to ensemble_strategy.py handle_event method:

def handle_event(self, event: Event):
    """Override to suppress signals during warmup."""
    if event.event_type == EventType.BAR:
        bar_data = event.data
        
        # Check if this is a warmup bar
        if bar_data.get('is_warmup', False):
            # Process bar for indicator updates but suppress signal generation
            self._update_indicators(bar_data)
            return  # Don't generate signals during warmup
        
    # Normal processing for test bars
    super().handle_event(event)
'''
    
    print("\nStrategy modification needed:")
    print(strategy_patch)

if __name__ == "__main__":
    # First, understand the warmup state
    simulate_indicator_warmup()
    
    print("\n" + "="*60)
    print("SOLUTION SUMMARY:")
    print("="*60)
    print("""
1. The optimizer warms up indicators on 798 training bars
2. At test start (bar 798), indicators are fully warmed
3. This allows immediate signal generation at 13:46:00

To match this in production:
- Option A: Modify data handler to process training bars first (warmup mode)
- Option B: Modify strategy to suppress signals during warmup
- Option C: Pre-calculate indicator states and initialize with them

The key is ensuring indicators have the exact same state at test start.
""")
    
    # Show what needs to match
    print("\nTarget signals to match:")
    print("1. BUY at 2024-03-28 13:46:00 (bar 798)")
    print("2. SELL at 2024-03-28 14:00:00 (bar 812)")
    print("... and 14 more signals")