#!/bin/bash
# Script to compare optimizer and production results using the correct Python environment

echo "========================================================================"
echo "COLD START FIX VERIFICATION"
echo "========================================================================"

# First, ensure we're using the config that has EnhancedOptimizerV2
echo ""
echo "1. Checking which optimizer is configured..."
echo "------------------------------------------------------------------------"
grep -E "enhanced_optimizer|optimizer.*type" config/config_debug_comparison.yaml | head -5

echo ""
echo "2. Running Optimizer (should use EnhancedOptimizerV2)..."
echo "------------------------------------------------------------------------"
echo "Command: python main.py --config config/config_debug_comparison.yaml --optimize-joint --log-level ERROR"
python main.py --config config/config_debug_comparison.yaml --optimize-joint --log-level ERROR 2>&1 | grep -E "Final.*value|Test metric:|Regimes detected|ADAPTIVE TEST|portfolio value" | tail -10

echo ""
echo "3. Running Production Backtest..."
echo "------------------------------------------------------------------------"
echo "Command: python run_production_backtest_v2.py --config config/config_debug_comparison.yaml --dataset test --log-level ERROR"
python run_production_backtest_v2.py --config config/config_debug_comparison.yaml --dataset test --log-level ERROR 2>&1 | grep -E "Final.*Value|PERFORMANCE BY REGIME" -A 5

echo ""
echo "========================================================================"
echo "NOTE: If values still don't match, it's because:"
echo "1. The optimizer is not using EnhancedOptimizerV2 (check config)"
echo "2. The RegimeDetector has different warmup periods"
echo "========================================================================"