#!/bin/bash
# Test the production runner with BacktestEngine

echo "Testing production runner with BacktestEngine..."
echo "=============================================="
echo ""

# First check if we have the required config file
if [ -f "config/config.yaml" ]; then
    echo "✓ Found config/config.yaml"
    
    # Run with ensemble strategy on full dataset
    echo ""
    echo "Running ensemble strategy backtest on full dataset..."
    echo "Command: python3 run_production_backtest_v2.py --config config/config.yaml --strategy ensemble --dataset full"
    echo ""
    
    python3 run_production_backtest_v2.py --config config/config.yaml --strategy ensemble --dataset full --log-level INFO
else
    echo "✗ config/config.yaml not found!"
    echo "Please ensure you have a valid configuration file."
fi