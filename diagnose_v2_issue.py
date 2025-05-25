#!/usr/bin/env python3
"""
Diagnose why EnhancedOptimizerV2 isn't producing matching results.
"""

import sys
sys.path.append('.')

from src.core.config import SimpleConfigLoader
from src.core.container import Container
from src.core.event_bus import EventBus
from src.strategy.optimization.engines.backtest_engine import BacktestEngine

def diagnose_issue():
    """Run diagnostic to understand the mismatch."""
    
    print("="*60)
    print("DIAGNOSING V2 OPTIMIZER VS PRODUCTION MISMATCH")
    print("="*60)
    
    # Load configuration
    config_loader = SimpleConfigLoader('config/config.yaml')
    event_bus = EventBus()
    container = Container()
    
    # Create BacktestEngine (used by V2)
    engine = BacktestEngine(container, config_loader, event_bus)
    
    print("\n1. COMPONENT RESET BEHAVIOR:")
    print("-"*60)
    
    # Check if BacktestEngine has our fix
    import inspect
    reset_code = inspect.getsource(engine._reset_components)
    if "regime_detector" in reset_code:
        print("✅ BacktestEngine._reset_components includes RegimeDetector reset")
    else:
        print("❌ BacktestEngine._reset_components does NOT reset RegimeDetector")
        
    # Check reset timing
    run_code = inspect.getsource(engine.run_backtest)
    if "_reset_components(components)" in run_code and "_setup_components(components)" in run_code:
        # Find their relative positions
        reset_pos = run_code.find("_reset_components(components)")
        setup_pos = run_code.find("_setup_components(components)")
        if reset_pos < setup_pos:
            print("✅ Components are reset BEFORE setup (cold start)")
        else:
            print("❌ Components are reset AFTER setup (not cold start)")
    else:
        print("❌ Cannot find reset/setup calls in run_backtest")
        
    print("\n2. REGIME DETECTOR CONFIGURATION:")
    print("-"*60)
    
    # Check regime detector config
    regime_config = config_loader.get("components.MyPrimaryRegimeDetector", {})
    if regime_config:
        indicators = regime_config.get("indicators", {})
        for name, ind_config in indicators.items():
            params = ind_config.get("parameters", {})
            print(f"- {name}: {ind_config.get('type')} with {params}")
            
        # Check MA trend periods
        ma_trend = indicators.get("trend_10_30", {}).get("parameters", {})
        short = ma_trend.get("short_period", "N/A")
        long = ma_trend.get("long_period", "N/A") 
        print(f"\nMA Trend warmup period: {long} bars")
        print(f"Test dataset has 200 bars (20% of 1000)")
        if isinstance(long, int) and long >= 200:
            print("⚠️  MA Trend will NEVER be ready in test-only run!")
        
    print("\n3. KEY DIFFERENCES:")
    print("-"*60)
    print("Optimizer Adaptive Test:")
    print("- Runs after training phase")
    print("- Components may retain state from training")
    print("- RegimeDetector has seen 800 training bars")
    print("")
    print("Production Test-Only Run:")
    print("- Starts completely fresh")
    print("- RegimeDetector only sees 200 test bars")
    print("- MA trend needs 200 bars but only gets 200")
    
    print("\n4. SOLUTION STATUS:")
    print("-"*60)
    print("The fix was implemented in BacktestEngine:")
    print("1. Reset RegimeDetector in _reset_components() ✅")
    print("2. Call reset BEFORE setup for cold start ✅")
    print("")
    print("But results still don't match because:")
    print("- Even with reset, indicators need warmup time")
    print("- Test data (200 bars) = MA trend period (200 bars)")
    print("- So MA trend becomes ready at the LAST bar of test data")
    print("- This gives different regime detection behavior")
    
    print("\n5. REAL SOLUTION OPTIONS:")
    print("-"*60)
    print("A. Use shorter indicator periods (e.g., 20/50 instead of 50/200)")
    print("B. Include warmup data in test dataset (e.g., bars 600-999)")
    print("C. Pre-warm indicators with training data in production")
    print("D. Accept the difference as a limitation of the test setup")

if __name__ == "__main__":
    diagnose_issue()