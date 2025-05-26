"""
Test the new strategy architecture.
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


def test_indicator():
    """Test indicator functionality."""
    print("\n=== Testing Indicators ===")
    
    # Test moving average
    ma = MovingAverageIndicator(name="test_ma", lookback_period=3)
    
    # Feed some data
    for i, price in enumerate([100, 102, 104, 103, 105]):
        bar_data = {'close': price, 'timestamp': f'2024-01-01T{i:02d}:00:00'}
        result = ma.update(bar_data)
        print(f"Price: {price}, MA ready: {result.ready}, MA value: {result.value}")
        
    # Test RSI
    print("\n--- Testing RSI ---")
    rsi = RSIIndicator(name="test_rsi", lookback_period=5)
    
    prices = [100, 102, 101, 103, 104, 102, 105, 107, 106, 108]
    for i, price in enumerate(prices):
        bar_data = {'close': price, 'timestamp': f'2024-01-01T{i:02d}:00:00'}
        result = rsi.update(bar_data)
        if result.ready:
            print(f"Price: {price}, RSI: {result.value:.2f}")
            

def test_rules():
    """Test rule functionality."""
    print("\n=== Testing Rules ===")
    
    # Create indicators
    fast_ma = MovingAverageIndicator(name="fast", lookback_period=2)
    slow_ma = MovingAverageIndicator(name="slow", lookback_period=3)
    
    # Create crossover rule
    rule = CrossoverRule(name="ma_cross")
    rule.add_dependency("fast_ma", fast_ma)
    rule.add_dependency("slow_ma", slow_ma)
    
    # Feed data and test signals
    prices = [100, 98, 102, 105, 103, 101, 99, 102, 105, 108]
    
    for i, price in enumerate(prices):
        bar_data = {'close': price, 'timestamp': f'2024-01-01T{i:02d}:00:00'}
        
        # Update indicators
        fast_ma.update(bar_data)
        slow_ma.update(bar_data)
        
        # Evaluate rule
        signal, strength = rule.evaluate(bar_data)
        
        if signal != 0:
            print(f"Bar {i}: Price={price}, Signal={signal}, Strength={strength:.2f}")
            

def test_parameter_space():
    """Test parameter space functionality."""
    print("\n=== Testing Parameter Space ===")
    
    from src.strategy.base import Parameter
    
    # Create parameter space
    space = ParameterSpace("test_space")
    
    # Add parameters
    space.add_parameter(
        Parameter(
            name='fast_period',
            param_type='discrete',
            values=[5, 10, 15, 20],
            default=10
        )
    )
    
    space.add_parameter(
        Parameter(
            name='slow_period',
            param_type='discrete',
            values=[20, 30, 40, 50],
            default=30
        )
    )
    
    space.add_parameter(
        Parameter(
            name='threshold',
            param_type='continuous',
            min_value=0.01,
            max_value=0.05,
            step=0.01,
            default=0.02
        )
    )
    
    # Sample parameters
    samples = space.sample(method='grid')
    
    print(f"Total parameter combinations: {len(samples)}")
    print("First 5 combinations:")
    for i, params in enumerate(samples[:5]):
        print(f"  {i+1}: {params}")
        

def test_composite_strategy():
    """Test composite strategy."""
    print("\n=== Testing Composite Strategy ===")
    
    # Create event bus
    event_bus = EventBus()
    
    # Create strategy
    strategy = CompositeMAStrategy(instance_name="test_strategy")
    
    # Initialize with config
    strategy.component_config = {
        'fast_ma_period': 5,
        'slow_ma_period': 10,
        'use_rsi_filter': True,
        'rsi_period': 14,
        'rsi_buy_threshold': 30,
        'rsi_sell_threshold': 70
    }
    
    # Set up context (minimal for testing)
    context = {
        'event_bus': event_bus,
        'container': None
    }
    
    # Initialize strategy
    strategy.initialize(context)
    strategy.start()
    
    # Subscribe to signals
    signals_received = []
    
    def on_signal(event):
        signals_received.append(event.payload)
        print(f"Signal received: {event.payload}")
        
    event_bus.subscribe(EventType.SIGNAL, on_signal)
    
    # Feed some test data
    prices = [100, 98, 96, 97, 99, 102, 105, 103, 101, 98, 95, 97, 100, 103, 106]
    
    for i, price in enumerate(prices):
        bar_event = Event(
            EventType.BAR,
            {
                'symbol': 'TEST',
                'close': price,
                'timestamp': f'2024-01-01T{i:02d}:00:00'
            }
        )
        event_bus.publish(bar_event)
        
    print(f"\nTotal signals generated: {len(signals_received)}")
    
    # Test parameter management
    print("\n--- Parameter Management ---")
    params = strategy.get_parameters()
    print(f"Current parameters: {params}")
    
    space = strategy.get_parameter_space()
    print(f"Parameter space: {space.to_dict()}")
    

def main():
    """Run all tests."""
    print("Testing New Strategy Architecture")
    print("=" * 50)
    
    test_indicator()
    test_rules()
    test_parameter_space()
    test_composite_strategy()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    

if __name__ == "__main__":
    main()