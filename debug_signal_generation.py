#!/usr/bin/env python3

"""
Test script to verify that weight changes actually affect signal generation.
This will simulate different scenarios to see if weights matter.
"""

import sys
import os
import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import SimpleConfigLoader
from src.core.event_bus import EventBus
from src.core.container import Container
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy
from src.core.event import Event, EventType

def test_signal_generation_with_weights():
    """Test if different weights produce different signals in conflict scenarios"""
    
    print("=== Testing Signal Generation with Different Weights ===")
    
    # Setup basic infrastructure
    config_loader = SimpleConfigLoader("config/config.yaml")
    event_bus = EventBus()
    container = Container()
    
    # Create strategy instance
    strategy = EnsembleSignalStrategy(
        instance_name="TestStrategy",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="components.ensemble_strategy",
        container=container
    )
    
    strategy.setup()
    strategy.start()
    
    # Test prices that should create a conflicting scenario
    # We'll create a scenario where MA gives one signal, RSI gives another
    
    print("\n1. Creating test scenario with conflicting MA and RSI signals")
    
    # First, let's feed some price data to build up the indicators
    base_price = 100.0
    prices = [
        95.0,   # Start low
        96.0, 97.0, 98.0, 99.0, 100.0,  # Gradual rise (potential MA buy signal)
        101.0, 102.0, 103.0, 104.0, 105.0,
        106.0, 107.0, 108.0, 109.0, 110.0,  # Continue rising (RSI might be overbought)
        111.0, 112.0, 113.0, 114.0, 115.0   # Strong rise (RSI definitely overbought)
    ]
    
    # Feed price data to both strategies with different weights
    print("\n2. Testing with MA-heavy weights (MA=0.8, RSI=0.2)")
    
    # Configure for MA-heavy
    strategy.set_parameters({
        "ma_rule.weight": 0.8,
        "rsi_rule.weight": 0.2,
        "short_window": 5,
        "long_window": 10
    })
    
    signals_ma_heavy = []
    
    # Capture signals
    def capture_signal_ma_heavy(event):
        if event.event_type == EventType.SIGNAL:
            signals_ma_heavy.append({
                'price': event.payload.get('price_at_signal'),
                'signal': event.payload.get('signal_type'),
                'reason': event.payload.get('reason', '')
            })
            print(f"   MA-heavy signal: {event.payload.get('signal_type')} at {event.payload.get('price_at_signal')} - {event.payload.get('reason', '')}")
    
    event_bus.subscribe(EventType.SIGNAL, capture_signal_ma_heavy)
    
    # Feed prices
    for i, price in enumerate(prices):
        timestamp = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=i)
        bar_event = Event(EventType.BAR, {
            'symbol': 'SPY',
            'close': price,
            'timestamp': timestamp,
            'open': price - 0.1,
            'high': price + 0.1,
            'low': price - 0.2,
            'volume': 1000
        })
        event_bus.publish(bar_event)
    
    event_bus.unsubscribe(EventType.SIGNAL, capture_signal_ma_heavy)
    
    print(f"\n   MA-heavy generated {len(signals_ma_heavy)} signals")
    
    # Reset strategy state
    strategy._current_signal_state = 0
    strategy._prices.clear()
    strategy._prev_short_ma = None
    strategy._prev_long_ma = None
    
    print("\n3. Testing with RSI-heavy weights (MA=0.2, RSI=0.8)")
    
    # Configure for RSI-heavy
    strategy.set_parameters({
        "ma_rule.weight": 0.2,
        "rsi_rule.weight": 0.8,
        "short_window": 5,
        "long_window": 10
    })
    
    signals_rsi_heavy = []
    
    # Capture signals
    def capture_signal_rsi_heavy(event):
        if event.event_type == EventType.SIGNAL:
            signals_rsi_heavy.append({
                'price': event.payload.get('price_at_signal'),
                'signal': event.payload.get('signal_type'),
                'reason': event.payload.get('reason', '')
            })
            print(f"   RSI-heavy signal: {event.payload.get('signal_type')} at {event.payload.get('price_at_signal')} - {event.payload.get('reason', '')}")
    
    event_bus.subscribe(EventType.SIGNAL, capture_signal_rsi_heavy)
    
    # Feed the same prices
    for i, price in enumerate(prices):
        timestamp = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=i)
        bar_event = Event(EventType.BAR, {
            'symbol': 'SPY',
            'close': price,
            'timestamp': timestamp,
            'open': price - 0.1,
            'high': price + 0.1,
            'low': price - 0.2,
            'volume': 1000
        })
        event_bus.publish(bar_event)
    
    event_bus.unsubscribe(EventType.SIGNAL, capture_signal_rsi_heavy)
    
    print(f"\n   RSI-heavy generated {len(signals_rsi_heavy)} signals")
    
    # Compare results
    print("\n4. Comparison:")
    
    if len(signals_ma_heavy) != len(signals_rsi_heavy):
        print(f"   ✅ Different number of signals: MA-heavy={len(signals_ma_heavy)}, RSI-heavy={len(signals_rsi_heavy)}")
        return True
    
    # Check if signals are different
    different_signals = False
    for i in range(min(len(signals_ma_heavy), len(signals_rsi_heavy))):
        ma_sig = signals_ma_heavy[i]
        rsi_sig = signals_rsi_heavy[i]
        
        if ma_sig['signal'] != rsi_sig['signal'] or abs(ma_sig['price'] - rsi_sig['price']) > 0.01:
            different_signals = True
            print(f"   Signal {i+1} differs:")
            print(f"     MA-heavy: {ma_sig['signal']} at {ma_sig['price']} - {ma_sig['reason']}")
            print(f"     RSI-heavy: {rsi_sig['signal']} at {rsi_sig['price']} - {rsi_sig['reason']}")
    
    if different_signals:
        print("   ✅ Weight changes produce different signals!")
        return True
    else:
        print("   ❌ Weight changes do NOT produce different signals")
        print("   This suggests the weights aren't affecting signal generation properly")
        
        # Debug: Print all signals
        print("\n   All MA-heavy signals:")
        for sig in signals_ma_heavy:
            print(f"     {sig}")
        print("\n   All RSI-heavy signals:")
        for sig in signals_rsi_heavy:
            print(f"     {sig}")
            
        return False

if __name__ == "__main__":
    test_signal_generation_with_weights()