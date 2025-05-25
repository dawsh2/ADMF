#!/bin/bash
# Run both options and compare results

echo "========================================================================"
echo "TESTING BOTH OPTIONS FOR MATCHING RESULTS"
echo "========================================================================"

echo -e "\nTarget Results:"
echo "- Optimizer Adaptive Test: \$100,058.98"
echo "- Production Test: \$99,870.04"
echo ""

# Option 1: Make production match optimizer
echo -e "\n1. OPTION 1: Production matching optimizer's data loading..."
echo "------------------------------------------------------------------------"
python production_match_optimizer.py 2>&1 | grep -E "Final Portfolio Value:|Regimes detected:|should match" | tail -5

# Option 2: Make optimizer match production  
echo -e "\n\n2. OPTION 2: Optimizer matching production's data loading..."
echo "------------------------------------------------------------------------"
python optimizer_match_production.py 2>&1 | grep -E "Final Portfolio Value:|PERFORMANCE BY REGIME|should match" -A 10 | tail -15

echo -e "\n========================================================================"
echo "CONCLUSION:"
echo "One of these approaches should produce matching results."
echo "The one that matches tells us which data loading approach is correct."
echo "========================================================================"