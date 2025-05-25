#!/usr/bin/env python3
"""
Diagnose why RegimeDetector behaves differently between OOS test and production.
"""

print("""
DIAGNOSIS: RegimeDetector Mismatch

The production run is detecting fewer regimes than the OOS test:
- OOS Test: default, trending_down, ranging_low_vol, trending_up_low_vol (4 regimes)
- Production: default, ranging_low_vol (2 regimes)

POSSIBLE CAUSES:

1. **Indicator Warmup Period** (Most Likely)
   - RegimeDetector indicators (RSI, ATR, MA) need warmup bars
   - OOS test might process more initial bars for warmup
   - Production might start with "cold" indicators
   
2. **Component Initialization Order**
   - OOS: RegimeDetector might be initialized before data processing
   - Production: Different initialization sequence
   
3. **Dataset Starting Point**
   - OOS test uses test portion after train/test split
   - Production might be using a different subset
   
4. **Indicator State Persistence**
   - OOS test might have warmed-up indicators from training
   - Production starts completely fresh

VERIFICATION STEPS:

1. Check the first 50 bars of regime detection in both runs
2. Log indicator values (RSI, ATR, trend) at start
3. Verify both use exact same test dataset
4. Track when each regime is first detected

To fix this, we need to ensure:
- Both runs start with same indicator state
- Both process same initial warmup bars
- RegimeDetector doesn't reset between train/test in optimizer
""")

print("\nRECOMMENDED ACTION:")
print("Add detailed logging to track RegimeDetector initialization and first 100 bars")
print("This will help identify exactly where the divergence begins.")