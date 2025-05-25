#!/bin/bash
# Run verification test to compare OOS and production results

echo "Running OOS vs Production Verification Test"
echo "=========================================="
echo ""
echo "This will:"
echo "1. Run the optimizer's adaptive test on test data"
echo "2. Run a standalone production backtest on the same test data"
echo "3. Compare the results to ensure they match"
echo ""
echo "Running verification..."
echo ""

python3 verify_oos_production_match.py