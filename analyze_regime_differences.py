#!/usr/bin/env python3
"""
Analyze why production has different regimes at the critical signal timestamps.
Check regime detection inputs and thresholds.
"""

import re
from datetime import datetime

def extract_regime_data(log_file, target_timestamps):
    """Extract regime detection data around target timestamps."""
    regime_data = {}
    regime_changes = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Extract regime changes
            regime_match = re.search(r"REGIME CHANGED: '(\w+)' → '(\w+)' at ([^']+)", line)
            if regime_match:
                regime_changes.append({
                    'from': regime_match.group(1),
                    'to': regime_match.group(2),
                    'timestamp_str': regime_match.group(3)
                })
            
            # Extract regime detector inputs if available
            detector_match = re.search(r'Regime detector.*trend=([^,]+), atr=([^,]+), rsi=([^)]+)', line)
            if detector_match:
                # Would need timestamp context
                pass
    
    return regime_data, regime_changes

def compare_regimes_at_signals():
    """Compare regime states at the two critical signal timestamps."""
    
    target_times = [
        datetime(2024, 3, 28, 13, 46, 0),
        datetime(2024, 3, 28, 14, 0, 0)
    ]
    
    print("REGIME ANALYSIS AT CRITICAL SIGNALS")
    print("="*70)
    
    # Check what we know from the previous analysis
    print("From previous analysis:")
    print("\nAt 13:46:00:")
    print("  Optimizer: Regime = default")
    print("  Production: Regime = trending_up_volatile")
    print("  → Production regime likely suppresses MA signals")
    
    print("\nAt 14:00:00:")
    print("  Optimizer: Regime = default") 
    print("  Production: Regime = trending_up_low_vol")
    print("  → Different regime affects signal generation")
    
    print("\n" + "="*70)
    print("REGIME THRESHOLD ANALYSIS")
    print("="*70)
    
    # The regime thresholds from config
    thresholds = {
        'trending_up_volatile': {
            'trend_10_30': {'min': 0.02},
            'atr_20': {'min': 0.15}
        },
        'trending_up_low_vol': {
            'trend_10_30': {'min': 0.02},
            'atr_20': {'max': 0.15}
        },
        'ranging_low_vol': {
            'trend_10_30': {'min': -0.01, 'max': 0.01},
            'atr_20': {'max': 0.12}
        },
        'trending_down': {
            'trend_10_30': {'max': -0.01}
        },
        'default': {
            # No specific thresholds - fallback regime
        }
    }
    
    print("Regime thresholds in use:")
    for regime, thresh in thresholds.items():
        print(f"  {regime}: {thresh}")
    
    print("\n" + "="*70)
    print("HYPOTHESIS")
    print("="*70)
    
    print("The issue is likely:")
    print("1. Production has more warmup data, affecting regime detector indicators")
    print("2. Different indicator histories lead to different trend/ATR values") 
    print("3. These different values trigger different regime classifications")
    print("4. Different regimes have different parameter sets that affect signal generation")
    
    print("\nOptimizer stays in 'default' regime because:")
    print("- Regime detector starts fresh with limited history")
    print("- Takes time to accumulate enough data for regime classification")
    print("- Uses default parameters during this period")
    
    print("\nProduction detects specific regimes because:")
    print("- Has longer warmup history for regime detection indicators")
    print("- Regime detector has more data to make confident classifications")
    print("- Applies regime-specific parameters immediately")
    
    print("\n" + "="*70)
    print("SOLUTIONS")
    print("="*70)
    
    print("Option 1: Match regime detector warmup")
    print("- Reset regime detector before test (like we proposed for optimizer)")
    print("- This would keep production in 'default' regime initially")
    
    print("\nOption 2: Use identical regime parameters")
    print("- Ensure all regimes use the same MA/RSI weights") 
    print("- This would make regime differences not affect signals")
    
    print("\nOption 3: Disable adaptive parameters")
    print("- Run production without regime-specific parameter changes")
    print("- Use fixed parameters throughout the test")

def main():
    compare_regimes_at_signals()
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    print("The cleanest solution is Option 1:")
    print("Reset the regime detector in production to match optimizer's fresh start.")
    print("\nThis can be done by:")
    print("1. Adding regime detector reset before test data processing")
    print("2. Or using a config that disables regime adaptation initially")
    print("3. Or ensuring regime detector starts in 'default' state")
    
    print("\nThis would:")
    print("✓ Keep production in 'default' regime like optimizer")
    print("✓ Use same weights (MA=0.2, RSI=0.8) → but we disabled RSI")
    print("✓ Generate signals at same timestamps")
    print("✓ Not require permanent code changes")

if __name__ == "__main__":
    main()