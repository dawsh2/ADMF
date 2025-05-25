#!/usr/bin/env python3
"""
Create a production config that matches optimizer's regime behavior.
Force production to use 'default' regime parameters consistently.
"""

def create_matching_config():
    """Create config that forces same regime parameters as optimizer default."""
    
    # The key insight: make ALL regimes use the same parameters as 'default'
    # This way, regime differences won't affect signal generation
    
    config = """# Production config that matches optimizer regime behavior
system:
  name: "ADMF-Production-Regime-Matched"
  version: "0.1.0"

logging:
  level: "INFO"

components:
  ensemble_strategy:
    symbol: "SPY"
    short_window_default: 10
    long_window_default: 20
    ma_rule.weight: 1.0  # Since we disabled RSI, use full MA weight
    rsi_indicator:
      period: 14
    rsi_rule:
      oversold_threshold: 30.0
      overbought_threshold: 70.0
      weight: 0.0  # Disabled to match effective optimizer behavior

  data_handler_csv:
    csv_file_path: "data/1000_1min.csv"
    symbol: "SPY"
    timestamp_column: "timestamp"
    train_test_split_ratio: 0.8  # Use same split as optimizer
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
    
    # CRITICAL: Disable regime detection or make all regimes identical
    # Option 1: Set very high thresholds so no regime changes occur
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
    
    # Option 1: Make thresholds impossible to meet (stay in default)
    regime_thresholds:
      trending_up_volatile:
        trend_10_30: {"min": 999.0}  # Impossible threshold
        atr_20: {"min": 999.0}
      trending_up_low_vol:
        trend_10_30: {"min": 999.0}
        atr_20: {"max": -999.0}  # Impossible threshold
      ranging_low_vol:
        trend_10_30: {"min": 999.0, "max": -999.0}  # Impossible range
        atr_20: {"max": -999.0}
      trending_down:
        trend_10_30: {"max": -999.0}  # Impossible threshold
      oversold_in_uptrend:
        rsi_14: {"max": -999.0}  # Impossible threshold
        trend_10_30: {"min": 999.0}

# Alternative Option 2: Make all regimes use identical parameters
# (Add this to strategy config section)
regime_parameters:
  default:
    short_window: 10
    long_window: 20
    rsi_indicator.period: 14
    rsi_rule.oversold_threshold: 30.0
    rsi_rule.overbought_threshold: 60.0
  trending_up_volatile:
    short_window: 10  # Same as default
    long_window: 20   # Same as default
    rsi_indicator.period: 14
    rsi_rule.oversold_threshold: 30.0
    rsi_rule.overbought_threshold: 60.0
  trending_up_low_vol:
    short_window: 10  # Same as default
    long_window: 20   # Same as default
    rsi_indicator.period: 14
    rsi_rule.oversold_threshold: 30.0
    rsi_rule.overbought_threshold: 60.0
  ranging_low_vol:
    short_window: 10  # Same as default
    long_window: 20   # Same as default
    rsi_indicator.period: 14
    rsi_rule.oversold_threshold: 30.0
    rsi_rule.overbought_threshold: 60.0
  trending_down:
    short_window: 10  # Same as default
    long_window: 20   # Same as default
    rsi_indicator.period: 14
    rsi_rule.oversold_threshold: 30.0
    rsi_rule.overbought_threshold: 60.0
"""
    
    config_path = 'config/config_regime_matched.yaml'
    with open(config_path, 'w') as f:
        f.write(config)
    
    return config_path

def create_simpler_config():
    """Create an even simpler config that just disables regime adaptation."""
    
    config = """# Simple config - disable regime changes entirely
system:
  name: "ADMF-Production-No-Regimes"
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
    train_test_split_ratio: 0.8
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

  # DISABLE REGIME DETECTOR ENTIRELY - comment out the whole section
  # MyPrimaryRegimeDetector:
  #   This forces the strategy to stay in 'default' mode always
"""
    
    config_path = 'config/config_no_regimes.yaml'
    with open(config_path, 'w') as f:
        f.write(config)
    
    return config_path

def main():
    print("CREATING REGIME-MATCHED CONFIGS")
    print("="*70)
    
    print("The problem:")
    print("- Optimizer stays in 'default' regime (regime detector starts fresh)")
    print("- Production detects specific regimes (longer warmup)")
    print("- Different regimes = different parameters = different signals")
    
    print("\nSolution 1: Force production to stay in 'default' regime")
    config1 = create_matching_config()
    print(f"✓ Created: {config1}")
    print("  Uses impossible thresholds to prevent regime changes")
    
    print("\nSolution 2: Disable regime detection entirely")  
    config2 = create_simpler_config()
    print(f"✓ Created: {config2}")
    print("  Comments out regime detector - stays in default mode")
    
    print(f"\nNext steps:")
    print(f"1. Test config 2 first (simpler): python main.py --config {config2}")
    print(f"2. If that doesn't work, try config 1: python main.py --config {config1}")
    print(f"3. Count signals and verify they match optimizer's 16 signals")
    print(f"4. If signals match, verify timestamps match exactly")

if __name__ == "__main__":
    main()