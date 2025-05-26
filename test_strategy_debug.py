"""
Debug why composite strategy isn't generating signals.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.strategy.base import (
    Strategy,
    MovingAverageIndicator,
    RSIIndicator,
    CrossoverRule,
    ThresholdRule,
    ParameterSpace
)
from src.strategy.implementations.composite_ma_strategy import CompositeMAStrategy
from src.core.event_bus import EventBus
from src.core.event import Event, EventType


def debug_composite_strategy():
    """Debug composite strategy signal generation."""
    print("=== Debugging Composite Strategy ===\n")
    
    # Create event bus
    event_bus = EventBus()
    
    # Create strategy
    strategy = CompositeMAStrategy(instance_name="debug_strategy")
    
    # Initialize with config - use shorter periods for faster signals
    config = {
        'fast_ma_period': 3,  # Shorter period
        'slow_ma_period': 5,  # Shorter period
        'use_rsi_filter': False,  # Disable RSI filter for now
        'aggregation_method': 'weighted'
    }
    strategy.component_config = config
    print(f"Strategy config: {config}")
    
    # Set up context
    context = {
        'event_bus': event_bus,
        'container': None
    }
    
    # Initialize strategy
    strategy.initialize(context)
    
    # Check what was actually created
    print(f"\nAfter initialization:")
    print(f"  Indicators: {list(strategy._indicators.keys())}")
    for name, ind in strategy._indicators.items():
        print(f"    {name}: lookback_period={ind.lookback_period}")
    print(f"  Rules: {list(strategy._rules.keys())}")
    print(f"  Use RSI filter: {config.get('use_rsi_filter')}")
    
    strategy.start()
    
    # Subscribe to signals
    signals_received = []
    
    def on_signal(event):
        signals_received.append(event.payload)
        print(f"\n🎯 SIGNAL RECEIVED: {event.payload}")
        
    event_bus.subscribe(EventType.SIGNAL, on_signal)
    
    # Create a price series that should trigger crossovers
    # Start low, trend up (bullish cross), then trend down (bearish cross)
    prices = [
        100, 99, 98,     # Initial downtrend
        99, 101, 103,    # Start uptrend (should trigger bullish crossover)
        105, 107, 108,   # Continue up
        107, 105, 103,   # Start downtrend (should trigger bearish crossover)
        101, 99, 97      # Continue down
    ]
    
    print("Feeding price data...")
    print("-" * 50)
    
    for i, price in enumerate(prices):
        print(f"\nBar {i}: Price = ${price}")
        
        # Create bar event
        bar_event = Event(
            EventType.BAR,
            {
                'symbol': 'TEST',
                'close': price,
                'open': price - 0.5,
                'high': price + 0.5,
                'low': price - 0.5,
                'volume': 1000000,
                'timestamp': f'2024-01-01T{i:02d}:00:00'
            }
        )
        
        # Publish event
        event_bus.publish(bar_event)
        
        # Check indicator states
        fast_ma = strategy._indicators.get('fast_ma')
        slow_ma = strategy._indicators.get('slow_ma')
        
        if fast_ma and slow_ma:
            print(f"  Fast MA ({fast_ma.lookback_period}): ready={fast_ma.ready}, value={fast_ma.value:.2f}" if fast_ma.value else "  Fast MA: not ready")
            print(f"  Slow MA ({slow_ma.lookback_period}): ready={slow_ma.ready}, value={slow_ma.value:.2f}" if slow_ma.value else "  Slow MA: not ready")
            
            # Check if crossover rule would trigger
            if fast_ma.ready and slow_ma.ready:
                diff = fast_ma.value - slow_ma.value
                print(f"  MA Difference: {diff:.2f}")
                
        # Check rules
        for rule_name, rule in strategy._rules.items():
            if hasattr(rule, '_last_signal') and rule._last_signal:
                print(f"  Rule '{rule_name}' last signal: {rule._last_signal}")
    
    print("\n" + "=" * 50)
    print(f"Total signals generated: {len(signals_received)}")
    
    # If no signals, let's check why
    if len(signals_received) == 0:
        print("\n⚠️  No signals generated. Checking components...")
        
        # Check indicators
        print("\nIndicators:")
        for name, indicator in strategy._indicators.items():
            print(f"  {name}: ready={indicator.ready}, value={indicator.value}")
            
        # Check rules
        print("\nRules:")
        for name, rule in strategy._rules.items():
            print(f"  {name}: ready={rule.ready}")
            
            # Check dependencies
            if hasattr(rule, '_dependencies'):
                print(f"    Dependencies: {list(rule._dependencies.keys())}")
                

def test_manual_components():
    """Test components manually to isolate issues."""
    print("\n\n=== Testing Components Manually ===\n")
    
    # Create indicators
    fast_ma = MovingAverageIndicator(name="fast", lookback_period=3)
    slow_ma = MovingAverageIndicator(name="slow", lookback_period=5)
    
    # Create rule
    rule = CrossoverRule(name="test_cross")
    rule.add_dependency("fast_ma", fast_ma)
    rule.add_dependency("slow_ma", slow_ma)
    
    # Same price series
    prices = [100, 99, 98, 99, 101, 103, 105, 107, 108]
    
    print("Manual component test:")
    print("-" * 50)
    
    for i, price in enumerate(prices):
        bar_data = {'close': price, 'timestamp': f'2024-01-01T{i:02d}:00:00'}
        
        # Update indicators
        fast_result = fast_ma.update(bar_data)
        slow_result = slow_ma.update(bar_data)
        
        print(f"\nBar {i}: Price=${price}")
        print(f"  Fast MA: ready={fast_result.ready}, value={fast_result.value:.2f}" if fast_result.ready else "  Fast MA: not ready")
        print(f"  Slow MA: ready={slow_result.ready}, value={slow_result.value:.2f}" if slow_result.ready else "  Slow MA: not ready")
        
        # Evaluate rule
        signal, strength = rule.evaluate(bar_data)
        if signal != 0:
            print(f"  📍 SIGNAL: {signal}, strength={strength:.2f}")
        else:
            print(f"  No signal")


def main():
    """Run debugging tests."""
    print("Strategy Signal Generation Debug")
    print("=" * 60)
    
    debug_composite_strategy()
    test_manual_components()
    

if __name__ == "__main__":
    main()