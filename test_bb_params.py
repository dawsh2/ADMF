#!/usr/bin/env python3
"""Test BB parameter combinations"""

import sys
sys.path.append('.')

from src.strategy.components.indicators.bollinger_bands import BollingerBandsIndicator
from src.strategy.components.rules.bollinger_bands_rule import BollingerBandsRule

# Create BB indicator and rule
bb_ind = BollingerBandsIndicator('bb_indicator')
bb_rule = BollingerBandsRule('bb_rule', bb_indicator=bb_ind)

# Get parameter spaces
bb_ind_space = bb_ind.get_parameter_space()
bb_rule_space = bb_rule.get_parameter_space()

# Sample individual spaces
ind_combos = bb_ind_space.sample()
rule_combos = bb_rule_space.sample()

print(f"BB Indicator parameter combinations: {len(ind_combos)}")
print(f"BB Rule parameter combinations (including indicator subspace): {len(rule_combos)}")
print(f"\nFirst few BB Rule combinations:")
for i, combo in enumerate(rule_combos[:5]):
    print(f"  {i+1}: {combo}")

# Check if the issue is in parameter naming
print(f"\nParameter names in rule space:")
print(f"  Direct parameters: {list(bb_rule_space._parameters.items())}")
print(f"  Subspaces: {list(bb_rule_space._subspaces.items())}")