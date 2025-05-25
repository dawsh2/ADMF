# Why Optimizer and Production Results Don't Match

## The Core Issue

Even with `EnhancedOptimizerV2` and the cold start fix, results don't match because:

### 1. Regime Detector Configuration (from config.yaml)
```yaml
trend_10_30:
  type: "simple_ma_trend"
  parameters: {"short_period": 5, "long_period": 20}
```

The MA trend indicator needs 20 bars to warm up.

### 2. Dataset Split
- Total data: 1000 bars
- Train: 800 bars (80%)
- Test: 200 bars (20%)

### 3. The Fundamental Problem

**In Optimizer's Adaptive Test:**
- RegimeDetector processes ALL 1000 bars during optimization
- Even with reset between train/test, the test still runs on bars 800-999
- MA trend (20 period) becomes ready quickly at bar 820
- Can detect multiple regimes: default, trending_down, ranging_low_vol, trending_up_low_vol
- Result: $100,058.98

**In Production Test-Only Run:**
- RegimeDetector starts fresh with ONLY test data
- Sees bars 0-199 (which are actually bars 800-999 of the full dataset)
- MA trend becomes ready at bar 20
- Detects fewer regimes: only default and ranging_low_vol
- Result: $99,870.04

## Why The Fix Didn't Work

The cold start fix ensures components reset between runs, but it can't change the fundamental data visibility issue:

1. **Optimizer**: Even with reset, still processes test data as bars 800-999
2. **Production**: Processes the same data as bars 0-199

The different bar indices mean:
- Different warmup behavior
- Different regime detection timing
- Different trading signals
- Different results

## Real Solutions

### Option 1: Change How Test Data Is Fed
Instead of using `set_active_dataset("test")`, the optimizer should:
1. Create a completely new data handler
2. Load ONLY the test portion of data
3. Process it from bar 0 (not bar 800)

### Option 2: Include Warmup Period
- Change test split to include warmup bars
- E.g., use bars 600-999 as "test" (400 bars)
- First 200 bars for warmup, last 200 for actual testing

### Option 3: Shorter Indicator Periods
- Change MA trend to use shorter periods that fit within test data
- But this changes the strategy's behavior

### Option 4: Container-Based Isolation
As you suggested, containerize train and test to ensure complete isolation:
- Separate data handlers for train and test
- Fresh component instances for each phase
- No shared state whatsoever

## The Simplest Fix

Modify `run_production_backtest_v2.py` to use the same data indexing as the optimizer:
- Load the full dataset
- Skip to bar 800
- Process bars 800-999

This would make production match the optimizer's view of the data.