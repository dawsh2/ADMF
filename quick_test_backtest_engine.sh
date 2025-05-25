#!/bin/bash
# Quick test script for BacktestEngine

echo "BacktestEngine Usage Examples"
echo "============================="
echo ""
echo "1. Run regime-adaptive backtest with optimized parameters:"
echo "   python3 run_production_backtest_v2.py \\"
echo "       --config config/config_adaptive_production.yaml \\"
echo "       --strategy regime_adaptive \\"
echo "       --adaptive-params regime_optimized_parameters.json \\"
echo "       --dataset test"
echo ""
echo "2. Run ensemble strategy backtest:"
echo "   python3 run_production_backtest_v2.py \\"
echo "       --config config/config.yaml \\"
echo "       --strategy ensemble \\"
echo "       --dataset full"
echo ""
echo "3. Verify consistency between optimizer and production:"
echo "   python3 verify_backtest_consistency.py"
echo ""
echo "Choose an option (1-3) or press Enter to exit: "
read choice

case $choice in
    1)
        echo "Running regime-adaptive backtest..."
        python3 run_production_backtest_v2.py \
            --config config/config_adaptive_production.yaml \
            --strategy regime_adaptive \
            --adaptive-params regime_optimized_parameters.json \
            --dataset test
        ;;
    2)
        echo "Running ensemble strategy backtest..."
        python3 run_production_backtest_v2.py \
            --config config/config.yaml \
            --strategy ensemble \
            --dataset full
        ;;
    3)
        echo "Verifying consistency..."
        python3 verify_backtest_consistency.py
        ;;
    *)
        echo "Exiting..."
        ;;
esac