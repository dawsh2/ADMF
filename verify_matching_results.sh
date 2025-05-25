#!/bin/bash
# Verify that optimizer OOS and production results match with the fix

echo "========================================================================"
echo "VERIFYING COLD START FIX - RESULTS SHOULD NOW MATCH"
echo "========================================================================"
echo ""
echo "Using EnhancedOptimizerV2 which includes the cold start fix"
echo ""

# Run optimizer with joint optimization (includes OOS test)
echo "1. Running Optimizer with --optimize-joint (includes OOS test)..."
echo "------------------------------------------------------------------------"
python main.py --config config/config.yaml --optimize-joint --log-level ERROR 2>&1 | tail -20 | grep -E "ADAPTIVE TEST|Final portfolio value:|Regimes detected:"

echo ""
echo "2. Running Production Backtest on test data..."
echo "------------------------------------------------------------------------"
python run_production_backtest_v2.py \
    --config config/config.yaml \
    --strategy regime_adaptive \
    --dataset test \
    --adaptive-params regime_optimized_parameters.json \
    --log-level ERROR 2>&1 | grep -E "Final Portfolio Value:|PERFORMANCE BY REGIME" -A 5

echo ""
echo "========================================================================"
echo "EXPECTED: Both should show similar final values now that V2 is being used"
echo "========================================================================"