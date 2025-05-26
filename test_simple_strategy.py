"""
Test a simple MA crossover strategy without RSI.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.strategy.base import Strategy, MovingAverageIndicator, CrossoverRule
from src.strategy.base.rules.crossover import TrueCrossoverRule
from src.core.event_bus import EventBus
from src.core.event import Event, EventType


class SimpleMAStrategy(Strategy):
    """Simple MA crossover strategy for testing."""
    
    def _initialize(self):
        """Initialize strategy components."""
        # Create indicators with short periods
        fast_ma = MovingAverageIndicator(name="fast_ma", lookback_period=3)
        slow_ma = MovingAverageIndicator(name="slow_ma", lookback_period=5)
        
        # Add indicators
        self.add_indicator("fast_ma", fast_ma)
        self.add_indicator("slow_ma", slow_ma)
        
        # Create crossover rule - use TrueCrossoverRule for actual crossovers only
        crossover_rule = TrueCrossoverRule(name="ma_crossover")
        crossover_rule.add_dependency("fast_ma", fast_ma)
        crossover_rule.add_dependency("slow_ma", slow_ma)
        
        # Add rule
        self.add_rule("ma_crossover", crossover_rule, weight=1.0)
        
        # Call parent initialization
        super()._initialize()
        
        print(f"SimpleMAStrategy initialized:")
        print(f"  Fast MA period: {fast_ma.lookback_period}")
        print(f"  Slow MA period: {slow_ma.lookback_period}")
    


def test_simple_strategy():
    """Test simple MA strategy."""
    print("=== Testing Simple MA Strategy ===\n")
    
    # Create event bus
    event_bus = EventBus()
    
    # Create strategy
    strategy = SimpleMAStrategy(instance_name="simple_ma")
    
    # Set up context
    context = {
        'event_bus': event_bus,
        'container': None
    }
    
    # Initialize and start
    strategy.initialize(context)
    strategy.start()
    
    # Subscribe to signals
    signals_received = []
    
    def on_signal(event):
        signal = event.payload
        signals_received.append(signal)
        print(f"\n🎯 SIGNAL: direction={signal['signal']}, "
              f"price=${signal['price']:.2f}, "
              f"strategy={signal['strategy']}")
        
    event_bus.subscribe(EventType.SIGNAL, on_signal)
    
    # Test with trending prices
    prices = [
        100, 99, 98,      # Initial values
        99, 101, 103,     # Uptrend - should trigger buy
        105, 106, 107,    # Continue up
        106, 104, 102,    # Downtrend - should trigger sell
        100, 98, 96       # Continue down
    ]
    
    print("\nFeeding price data:")
    print("-" * 50)
    
    for i, price in enumerate(prices):
        # Create bar event
        bar_event = Event(
            EventType.BAR,
            {
                'symbol': 'TEST',
                'close': float(price),
                'timestamp': f'2024-01-01T{i:02d}:00:00'
            }
        )
        
        # Publish event
        event_bus.publish(bar_event)
        
        # Show indicator states
        fast_ma = strategy._indicators['fast_ma']
        slow_ma = strategy._indicators['slow_ma']
        
        print(f"\nBar {i}: Price=${price}")
        if fast_ma.ready:
            print(f"  Fast MA: {fast_ma.value:.2f}")
        else:
            print(f"  Fast MA: not ready")
            
        if slow_ma.ready:
            print(f"  Slow MA: {slow_ma.value:.2f}")
        else:
            print(f"  Slow MA: not ready")
            
        if fast_ma.ready and slow_ma.ready:
            diff = fast_ma.value - slow_ma.value
            print(f"  Difference: {diff:+.2f}")
    
    print("\n" + "=" * 50)
    print(f"Total signals generated: {len(signals_received)}")
    

if __name__ == "__main__":
    test_simple_strategy()