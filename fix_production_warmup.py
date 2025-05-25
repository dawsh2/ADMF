#!/usr/bin/env python3
"""
Fix production to match optimizer by pre-warming indicators.

Solution 1: Add a warmup period before the test data
Solution 2: Force indicators to reset between train/test in optimizer
"""

def create_warmup_config():
    """Create a config that includes warmup bars before test data."""
    config = """# Adaptive Production configuration with indicator warmup
system:
  name: "ADMF-Adaptive-Production-NoRSI-Warmup"
  version: "0.1.0"

logging:
  level: "INFO"

components:
  ensemble_strategy:
    symbol: "SPY"
    short_window_default: 10
    long_window_default: 20
    ma_rule.weight: 1.0
    rsi_indicator:
      period: 14
    rsi_rule:
      oversold_threshold: 30.0
      overbought_threshold: 70.0
      weight: 0.0

  data_handler_csv:
    csv_file_path: "data/1000_1min.csv"
    symbol: "SPY"
    timestamp_column: "timestamp"
    train_test_split_ratio: 0.80
    # Add warmup configuration
    warmup_bars: 20  # Process 20 bars before test to warm up indicators
    open_column: "Open"
    high_column: "High"
    low_column: "Low"
    close_column: "Close"
    volume_column: "Volume"

  basic_portfolio:
    initial_cash: 100000.00

  basic_risk_manager:
    target_trade_quantity: 100

  simulated_execution_handler:
    default_quantity: 100
    commission_per_trade: 0.005
    commission_type: "per_share"
    passthrough: false
    fill_price_logic: "signal_price"

  MyPrimaryRegimeDetector:
    min_regime_duration: 2
    verbose_logging: false
    summary_interval: 50
    indicators:
      rsi_14:
        type: "rsi"
        parameters: {"period": 14}
      atr_20:
        type: "atr"
        parameters: {"period": 10}
      trend_10_30:
        type: "simple_ma_trend"
        parameters: {"short_period": 5, "long_period": 20}
    regime_thresholds:
      trending_up_volatile:
        trend_10_30: {"min": 0.02}
        atr_20: {"min": 0.15}
      trending_up_low_vol:
        trend_10_30: {"min": 0.02}
        atr_20: {"max": 0.15}
      ranging_low_vol:
        trend_10_30: {"min": -0.01, "max": 0.01}
        atr_20: {"max": 0.12}
      trending_down:
        trend_10_30: {"max": -0.01}
      oversold_in_uptrend:
        rsi_14: {"max": 40}
        trend_10_30: {"min": 0.01}
"""
    
    with open('config/config_adaptive_production_warmup.yaml', 'w') as f:
        f.write(config)
    print("Created config/config_adaptive_production_warmup.yaml")

def implement_warmup_in_main():
    """Show how to implement warmup in main.py"""
    code = '''
# In main.py, before running the backtest, add warmup logic:

def run_with_warmup(data_handler, strategy, warmup_bars=20):
    """Run warmup bars to pre-warm indicators before test."""
    # Get the data handler's test data
    if hasattr(data_handler, '_test_df') and data_handler._test_df is not None:
        test_df = data_handler._test_df
        
        # Get warmup data from training set
        if hasattr(data_handler, '_train_df') and data_handler._train_df is not None:
            train_df = data_handler._train_df
            
            # Take last warmup_bars from training data
            warmup_data = train_df.iloc[-warmup_bars:].copy()
            
            # Process warmup bars without generating signals
            print(f"Processing {warmup_bars} warmup bars to pre-warm indicators...")
            
            # Temporarily disable signal generation
            original_weights = {}
            if hasattr(strategy, 'ma_rule') and hasattr(strategy.ma_rule, 'weight'):
                original_weights['ma'] = strategy.ma_rule.weight
                strategy.ma_rule.weight = 0.0
            if hasattr(strategy, 'rsi_rule') and hasattr(strategy.rsi_rule, 'weight'):
                original_weights['rsi'] = strategy.rsi_rule.weight  
                strategy.rsi_rule.weight = 0.0
            
            # Process warmup bars
            for idx, row in warmup_data.iterrows():
                bar_event = Event(
                    event_type="BAR",
                    payload={
                        "symbol": "SPY",
                        "timestamp": row.name,
                        "open": row["Open"],
                        "high": row["High"],
                        "low": row["Low"],
                        "close": row["Close"],
                        "volume": row["Volume"]
                    }
                )
                # This will update indicators but not generate signals
                strategy.on_event(bar_event)
            
            # Restore original weights
            if 'ma' in original_weights:
                strategy.ma_rule.weight = original_weights['ma']
            if 'rsi' in original_weights:
                strategy.rsi_rule.weight = original_weights['rsi']
            
            print(f"Warmup complete. Indicators are now pre-warmed.")
'''
    print("\nImplementation for main.py:")
    print("="*70)
    print(code)

def alternative_fix_optimizer():
    """Show how to fix optimizer to NOT carry over indicator state."""
    code = '''
# In enhanced_optimizer.py, modify _run_regime_adaptive_test to reset indicators:

def _run_regime_adaptive_test(self, optimized_params_by_regime):
    """Run adaptive test with proper indicator reset."""
    
    # ... existing code ...
    
    # CRITICAL: Reset all indicators before test run
    # This ensures test starts with cold indicators like production
    
    # Reset strategy indicators
    if hasattr(strategy_to_optimize, '_indicators'):
        for indicator in strategy_to_optimize._indicators.values():
            if hasattr(indicator, 'reset'):
                indicator.reset()
    
    # Reset MA values
    if hasattr(strategy_to_optimize, '_short_ma_series'):
        strategy_to_optimize._short_ma_series = []
    if hasattr(strategy_to_optimize, '_long_ma_series'):
        strategy_to_optimize._long_ma_series = []
    
    # Reset RSI
    if hasattr(strategy_to_optimize, 'rsi_indicator'):
        strategy_to_optimize.rsi_indicator.reset()
    
    # Reset regime detector indicators
    if hasattr(regime_detector, '_indicators'):
        for indicator in regime_detector._indicators.values():
            if hasattr(indicator, 'reset'):
                indicator.reset()
    
    print("All indicators reset to cold state for test run")
    
    # Now run test with cold indicators...
'''
    print("\nAlternative Fix - Reset indicators in optimizer:")
    print("="*70)
    print(code)

def main():
    print("PROPOSED FIXES FOR PRODUCTION-OPTIMIZER MISMATCH")
    print("="*80)
    
    print("\nRoot Cause:")
    print("-----------")
    print("The optimizer runs on training data first, which warms up MA indicators.")
    print("When it switches to test data, indicators retain their warmed state.")
    print("Production starts with cold indicators on test data.")
    
    print("\n\nSOLUTION 1: Pre-warm Production Indicators")
    print("-"*50)
    create_warmup_config()
    implement_warmup_in_main()
    
    print("\n\nSOLUTION 2: Force Cold Start in Optimizer")
    print("-"*50)
    alternative_fix_optimizer()
    
    print("\n\nRECOMMENDATION:")
    print("-"*50)
    print("Solution 1 is preferred because:")
    print("1. It matches the optimizer's behavior exactly")
    print("2. It's more realistic - in real trading you'd have historical data")
    print("3. It's easier to implement without modifying the optimizer")
    print("\nTo implement: Add the warmup logic to main.py before starting the backtest.")

if __name__ == "__main__":
    main()