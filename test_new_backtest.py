#!/usr/bin/env python3
"""
Test script to run a backtest using the new component architecture.
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.bootstrap import Bootstrap, RunMode
from src.core.config import SimpleConfigLoader


def run_backtest():
    """Run a simple backtest with the new components."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create minimal config for backtest
    config = {
        'mode': 'backtest',
        'data': {
            'file_path': 'data/SPY_1min.csv',
            'symbol': 'SPY'
        },
        'components': {
            'data_handler_csv': {
                'file_path': 'data/SPY_1min.csv',
                'symbol': 'SPY'
            },
            'basic_portfolio': {
                'initial_cash': 100000,
                'position_limits': {
                    'max_position_size': 0.1,
                    'max_total_exposure': 0.8
                }
            },
            'basic_risk_manager': {
                'max_position_size': 0.1,
                'max_portfolio_heat': 0.06,
                'stop_loss_percent': 0.02
            },
            'simulated_execution_handler': {
                'slippage': 0.0001,
                'commission': 0.001
            },
            'regime_adaptive_strategy': {
                'regime_params_file': 'adaptive_regime_parameters.json',
                'regime_detector_key': 'MyPrimaryRegimeDetector'
            },
            'MyPrimaryRegimeDetector': {
                'min_regime_duration': 5,
                'verbose_logging': False
            }
        }
    }
    
    print("Creating Bootstrap...")
    bootstrap = Bootstrap()
    
    try:
        print("Initializing system...")
        # Initialize with backtest mode
        context = bootstrap.initialize(
            config=config,
            run_mode=RunMode.BACKTEST
        )
        
        print("Starting components...")
        bootstrap.start_all()
        
        print("\nComponents initialized:")
        for name, component in bootstrap.components.items():
            print(f"  - {name}: {component.__class__.__name__} (state: {component.state})")
        
        print("\nRunning backtest...")
        # Get data handler to drive the backtest
        data_handler = bootstrap.components.get('data_handler')
        if data_handler:
            # Run for first 100 bars as a test
            bar_count = 0
            max_bars = 100
            
            print(f"Processing first {max_bars} bars...")
            while data_handler.has_data() and bar_count < max_bars:
                data_handler.update_bars()
                bar_count += 1
                
                if bar_count % 20 == 0:
                    print(f"  Processed {bar_count} bars...")
            
            print(f"\nBacktest completed: {bar_count} bars processed")
            
            # Get portfolio status
            portfolio = bootstrap.components.get('portfolio')
            if portfolio:
                status = portfolio.get_status()
                print(f"\nPortfolio Status:")
                print(f"  Cash: ${status.get('cash', 0):,.2f}")
                print(f"  Positions: {status.get('position_count', 0)}")
                print(f"  Total Value: ${status.get('total_value', 0):,.2f}")
        
    except Exception as e:
        print(f"\nError during backtest: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\nStopping components...")
        bootstrap.stop_all()
        
        print("Cleaning up...")
        bootstrap.teardown()
        
        print("\nBacktest complete!")


if __name__ == "__main__":
    run_backtest()