#!/usr/bin/env python3
"""
Direct test of refactored components without Bootstrap.
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.core.component_base import ComponentBase, ComponentState
from src.core.event_bus import EventBus
from src.core.container import Container
from src.data.csv_data_handler import CSVDataHandler
from src.risk.basic_portfolio import BasicPortfolio
from src.strategy.regime_adaptive_strategy import RegimeAdaptiveStrategy


def test_components():
    """Test components directly."""
    print("Testing Refactored Components")
    print("=" * 50)
    
    # Create infrastructure
    event_bus = EventBus()
    container = Container()
    
    # Mock config - note the keys match what the components expect
    config = {
        'components': {
            'data_handler_csv': {
                'csv_file_path': 'data/SPY_1min.csv',  # Changed from 'file_path'
                'symbol': 'SPY',
                'timestamp_column': 'timestamp'
            },
            'basic_portfolio': {
                'initial_cash': 100000
            },
            'regime_adaptive_strategy': {
                'symbol': 'SPY',
                'regime_params_file': 'adaptive_regime_parameters.json'
            }
        }
    }
    
    try:
        print("\n1. Testing CSVDataHandler...")
        data_handler = CSVDataHandler('data_handler', 'components.data_handler_csv')
        print(f"   Created: state={data_handler.state}")
        
        context = {
            'config': config,
            'event_bus': event_bus,
            'container': container
        }
        data_handler.initialize(context)
        print(f"   Initialized: state={data_handler.state}")
        
        print("\n2. Testing BasicPortfolio...")
        portfolio = BasicPortfolio('portfolio', 'components.basic_portfolio')
        print(f"   Created: state={portfolio.state}")
        
        portfolio.initialize(context)
        print(f"   Initialized: state={portfolio.state}")
        container.register('portfolio', portfolio)
        
        print("\n3. Testing RegimeAdaptiveStrategy...")
        strategy = RegimeAdaptiveStrategy('strategy', 'components.regime_adaptive_strategy')
        print(f"   Created: state={strategy.state}")
        
        # Strategy needs data_handler and portfolio
        container.register('data_handler', data_handler)
        strategy.initialize(context)
        print(f"   Initialized: state={strategy.state}")
        
        print("\n4. Starting components...")
        data_handler.start()
        print(f"   DataHandler started: state={data_handler.state}")
        
        portfolio.start()
        print(f"   Portfolio started: state={portfolio.state}")
        
        strategy.start()
        print(f"   Strategy started: state={strategy.state}")
        
        print("\n5. Processing a few bars...")
        for i in range(5):
            if data_handler.has_data():
                data_handler.update_bars()
                print(f"   Processed bar {i+1}")
            else:
                print(f"   No more data after {i} bars")
                break
                
        print("\n6. Getting portfolio status...")
        status = portfolio.get_status()
        print(f"   Cash: ${status.get('cash', 0):,.2f}")
        print(f"   Positions: {status.get('position_count', 0)}")
        
        print("\n7. Stopping components...")
        strategy.stop()
        portfolio.stop()
        data_handler.stop()
        print("   All components stopped")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(test_components())