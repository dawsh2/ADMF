#!/bin/bash
# Verify both runs and compare results

echo "========================================================================"
echo "COMPARING OPTIMIZER VS PRODUCTION RESULTS"
echo "========================================================================"

# First run optimizer
echo -e "\n1. Running Optimizer with --optimize-joint..."
echo "------------------------------------------------------------------------"
python main.py --config config/config.yaml --optimize-joint --log-level ERROR 2>&1 | grep -E "Final portfolio value:|Regimes detected:" | tail -2

# Then run production with matching setup
echo -e "\n\n2. Running Production with Matching Setup..."
echo "------------------------------------------------------------------------"
python run_production_matching_optimizer.py \
    --config config/config.yaml \
    --adaptive-params regime_optimized_parameters.json \
    --log-level ERROR 2>&1 | grep -E "Final Portfolio Value:|PERFORMANCE BY REGIME" -A 10

echo -e "\n========================================================================"
echo "ANALYSIS:"
echo "If results don't match, let's check what's different..."
echo "========================================================================"

# Check if regime parameters file exists and show content
echo -e "\nRegime parameters being used:"
if [ -f regime_optimized_parameters.json ]; then
    echo "File exists. Content:"
    cat regime_optimized_parameters.json | python -m json.tool | head -20
else
    echo "ERROR: regime_optimized_parameters.json not found!"
fi