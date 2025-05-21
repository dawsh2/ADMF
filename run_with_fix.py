#!/usr/bin/env python3
"""
Run the optimization with fixes for:
1. Component dependencies error in EnhancedOptimizer
2. Enhanced on_classification_change in EnsembleSignalStrategy
"""

import sys
import os

print("Applying fixes...")

# Import and apply the fix for component_dependencies error
from fix_component_dependencies_error import fix_enhanced_optimizer
if not fix_enhanced_optimizer():
    print("Failed to apply fix for component_dependencies error")
    sys.exit(1)
    
print("Fix applied successfully. Running optimization...")

# Run the main script with optimization (without debug flag)
os.system("python main.py --config config/config.yaml --optimize --bars 1000")