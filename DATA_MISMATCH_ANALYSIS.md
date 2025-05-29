# Data Mismatch Analysis

## Critical Finding: Different Data Windows

The optimization and independent test runs are using **different 200-bar windows** from the dataset!

### Evidence

#### Optimization Test Phase
- First regime change: 2024-03-28 **16:20:00**
- MA values: fast=523.3180, slow=523.3545
- RSI: 37-46 range

#### Independent Test Run  
- First regime change: 2024-03-28 **14:17:00**
- MA values: fast=523.2025, slow=523.3240
- RSI: 37-59 range

**Time difference: ~2 hours earlier data in independent test!**

## Root Cause

1. **Optimization**: Uses bars 801-1000 (last 200 of 1000) as test set
2. **Independent Test**: Appears to use a different 200-bar window (possibly first 200 of test split)

## Impact

- Different price data → Different indicator values
- Different indicator values → Different trading signals  
- Different signals → Different trades (20 vs 18)
- Different trades → Different returns (-0.03% vs -0.07%)

## Solution Required

Ensure `--dataset test` uses the **exact same bars** as the optimization test phase:
- Should be bars 801-1000 from the 1000-bar dataset
- Same timestamps (16:XX time range)
- Same price values

## Verification

To confirm this is the issue, we need to log:
1. First and last timestamp of test data in both runs
2. Bar indices being used
3. Ensure train/test split is consistent