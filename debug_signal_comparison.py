#!/usr/bin/env python3
"""Compare signal generation between optimization and production"""

import sys
sys.path.append('/Users/daws/ADMF')

# Check the key differences
print("=" * 80)
print("KEY DIFFERENCES BETWEEN OPTIMIZATION AND PRODUCTION")
print("=" * 80)

print("\n1. OPTIMIZATION ADAPTIVE TEST:")
print("   - Final value: 99,396.03")
print("   - Trade count: 234") 
print("   - Uses regime switching with adaptive parameters")
print("   - Weights: 0.341/0.659 (hard-coded in ensemble strategy)")

print("\n2. PRODUCTION RUN:")
print("   - Final value: 100,045.42")
print("   - Trade segments: 150")
print("   - Also uses regime switching")
print("   - Weights: NOW 0.5/0.5 (after our change)")

print("\n3. POSSIBLE EXPLANATIONS:")
print("   a) Different weights (0.341/0.659 vs 0.5/0.5)")
print("   b) Trade segment counting vs actual trades")
print("   c) Different initialization or timing")
print("   d) Production might be applying parameters differently")

print("\n4. THE ISSUE:")
print("   - Optimization shows adaptive test got 99,396.03")
print("   - But production with same parameters gets 100,045.42")
print("   - This suggests production is NOT matching optimization behavior")

print("\n5. WHAT TO CHECK:")
print("   - Are both runs starting with the same initial regime?")
print("   - Are regime changes happening at the same times?")
print("   - Are the parameters being applied identically?")
print("   - Is the trade execution logic the same?")