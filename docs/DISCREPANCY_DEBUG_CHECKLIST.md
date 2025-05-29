# Discrepancy Debug Checklist

## Objective
Ensure that optimization test phase and independent test run (`--dataset test`) produce identical results.

## Current Status
- **Optimization Test Phase**: -0.04% return, 18 trades (SELL first)
- **Independent Test Run**: -0.07% return, 19 trades (BUY first)
- **Key Discrepancy**: Opposite first signals despite same data!

## 🚨 NEW ROOT CAUSE DISCOVERED - INDICATOR BUFFERING 🚨
**Critical Finding**: Indicators retain buffered data despite reset!
- **Optimization**: MA indicators have 100 bars buffered when reset is called
- **Test Run**: MA indicators have 0 bars buffered (true cold start)
- **Result**: Different indicator states → Opposite signals from bar 1!

### Evidence:
1. **Bar 5 (13:52:00)** - Same parameters (MA 5/20) but:
   - Optimization: First 11 signals are SELL (-1)
   - Test Run: First 4 signals are BUY (+1)
   
2. **Reset Log Messages**:
   - Optimization: "RESET - clearing 100 bars of history"
   - Test Run: "RESET - clearing 0 bars of history"

3. **Missing Indicators**: Only MA and RSI show reset logs
   - BB (Bollinger Bands) indicator not logging reset
   - MACD indicator not logging reset
   - But they ARE being created and used with significant weights

### Root Cause:
The indicator reset() method clears buffers, but in optimization the indicators already have 100 bars of training data buffered. This buffered data affects calculations even after reset, causing different signals.

### Impact:
- Different MA crossover signals
- Different RSI values
- Different BB and MACD signals
- All rules produce different outputs → Opposite trading signals

## Checklist

### 1. Date Ranges and Datasets ❌ CRITICAL BUG - WRONG DATA!
**Status**: COMPLETELY DIFFERENT DATA BEING USED
- Test run: 2024-03-28 13:48:00 to 17:07:00 (indices 800-999) ✓
- Optimization: 2024-03-26 14:50:00 to 15:09:00 (indices 80-99) ❌

**Evidence**:
- Test run MA prices: [523.39, 523.35, 523.41...] (around $523)
- Optimization MA prices: [521.01, 521.07, 521.22...] (around $521)
- These are from DIFFERENT DAYS!

**Root Cause**: Index calculation error
- Should use indices 800-999 for 20% test split of 1000 bars
- Actually using indices 80-99 (decimal place error!)
- Only 20 bars instead of 200 bars

**Impact**: This explains EVERYTHING
- Different price data → Different indicator values
- Different indicators → Different regime classifications  
- Different regimes → Different signals and trades

### 2. Rules and Configuration ❌ NOT VERIFIED
**Status**: Need to verify exact rule instantiation
- [ ] Verify same rules are loaded (MA, RSI, BB, MACD)
- [ ] Verify rule initialization order
- [ ] Verify rule weights at initialization
- [ ] Verify rule-specific configurations
- [ ] Check for rule instance IDs to ensure no duplicate rules

**Next Steps**:
- Log all rules at strategy initialization
- Log rule parameters and weights
- Verify no extra rules from config inheritance

### 3. Regime Classifier Configs ❌ CRITICAL ISSUE
**Status**: Multiple regime detectors detected!
- [x] Found: `MyPrimaryRegimeDetector` (always returns 'default')
- [x] Found: `regime_detector` (properly configured)
- [ ] Verify which detector the strategy listens to
- [ ] Verify detector initialization parameters
- [ ] Verify indicator configuration within detectors

**Key Issue**: Strategy may be listening to wrong detector
**Next Steps**:
- Implement detector name filtering
- Remove or disable `MyPrimaryRegimeDetector`
- Verify event subscription order

### 4. Risk Configs ❌ NOT VERIFIED
**Status**: Not yet checked
- [ ] Position sizing configuration
- [ ] Max positions limit
- [ ] Risk per trade settings
- [ ] Stop loss/take profit settings
- [ ] Minimum position size requirements

**Next Steps**:
- Log all risk manager parameters
- Verify portfolio manager configuration

### 5. Parameters ✅ PARTIALLY COMPLETE
**Status**: Parameters loaded from same file but application differs
- [x] Both load from `test_regime_parameters.json`
- [ ] Verify parameter application timing
- [ ] Verify parameter override mechanisms
- [ ] Check for parameter caching issues
- [ ] Verify indicator parameter updates

**Next Steps**:
- Log every parameter change with timestamp
- Verify parameters are applied before processing bars

### 6. Regime Change Timestamps ❌ MISMATCH
**Status**: Different number of regime changes (38 vs 34)
- [ ] Log exact timestamp of each regime change
- [ ] Log trigger values (trend, volatility, RSI)
- [ ] Verify regime change cooldown period
- [ ] Check for race conditions in event ordering

**Next Steps**:
- Create detailed regime change log with trigger values
- Compare warmup behavior between runs

### 7. Execution Path ❌ PARTIALLY VERIFIED
**Status**: Some milestones match, others don't
- [x] "RUNNING TEST DATASET WITH OPTIMIZED PARAMETERS" ✓
- [ ] "ENABLED regime switching" missing in test run
- [x] "Applied rule parameter" counts differ (494 vs 442)
- [ ] "Portfolio reset" missing in test run

**Next Steps**:
- Trace complete execution flow
- Verify component initialization order

### 8. Signal History ⚠️ CONCERNING
**Status**: 0 signals in both runs!
- [ ] Verify rules are being evaluated
- [ ] Check signal generation logic
- [ ] Verify signal aggregation
- [ ] Check for signal filtering/suppression

**Critical Issue**: No trades because no signals generated
**Next Steps**:
- Add detailed rule evaluation logging
- Check indicator readiness before evaluation
- Verify weight application

### 9. Trade History ⚠️ BLOCKED BY SIGNALS
**Status**: 0 trades due to 0 signals
- [ ] Cannot verify until signals are generated
- [ ] Position sizing verification pending
- [ ] Order execution verification pending

**Blocked by**: Signal generation issue

### 10. Regime Change History ❌ EXPLAINED BY ROOT CAUSE
**Status**: Differences now make sense!
- Optimization: 38 changes, starts with "default -> trending_up"
- Test run: 34 changes, starts with "default -> trending_down"

**Now Explained**: They're analyzing completely different market data!
- March 26 data (optimization) had different price movements than March 28 (test)
- Of course the regime changes would be different!

### 11. Indicator Resets ❌ CRITICAL ISSUE - ROOT CAUSE FOUND
**Status**: Major discrepancy in indicator states
- [x] Optimization: Indicators warm from 800 training bars
- [x] Test run: Indicators start cold
- [x] MA values differ significantly after regime changes
- [ ] Need consistent warmup behavior

**Current Pursuit**: 
- Just commented out regime detector reset in workflow_orchestrator.py
- ~~Hypothesis: Optimization test phase should keep warm indicators from training~~
- **REALIZATION**: Independent test run has NO training phase to warm up from!
- **NEW APPROACH NEEDED**: Either:
  a) Cold start for BOTH (reset indicators in optimization test phase), OR
  b) Warm start for BOTH (save indicator state from training, restore for test run)

**Critical Issue**: Independent test run starts fresh with no prior data
- Cannot replicate warm indicators without saving/restoring state
- This explains the 38 vs 34 regime change difference

**Actions Taken**:
- [x] REVERTED the commenting out (uncommented the reset code)
- [x] Both runs now use cold start behavior
- [x] Tested - still getting 38 vs 34 regime changes

**Test Results**:
- ✅ ENABLED regime switching now shows for both! 
- ❌ Still have regime change mismatch (38 vs 34)
- ❌ Returns still differ (-0.02% vs -0.07%)

**Key Understanding**:
- Indicators should start cold for train/test (no leakage) ✓
- Indicators should NOT reset during regime changes ✓
- Need buffers for different lookback periods ✓

**New Hypothesis**: The MyPrimaryRegimeDetector might still be interfering

**Investigation Results**:
- [x] MyPrimaryRegimeDetector publishes 1 CLASSIFICATION event
- [x] This happens BEFORE regime_detector starts
- [x] First divergence: MyPrimary says 'default', real detector would say 'trending_down'
- [x] Added regime_detector_name filter to strategy config

**Ready to Test Again**: Should now ignore MyPrimaryRegimeDetector events

### Critical Discovery: Opposite First Classification
**Finding**: First regime change is COMPLETELY OPPOSITE
- Optimization: default -> trending_up
- Test run: default -> trending_down

**Timing Discovery - CRITICAL BUG**: 
- All optimization changes happen at 17:18:29 (same second!)
- All test run changes happen at 17:18:31 (2 seconds later)
- **THIS IS IMPOSSIBLE WITH 1-MINUTE BAR DATA!**

**Key Insight**: We have 1-minute bars, so regime changes should be:
- At most one per minute
- At different timestamps (13:48, 13:49, etc.)
- NOT multiple changes in the same second!

**This indicates a fundamental bug**:
- The optimization test phase is processing ALL bars instantly
- Multiple BAR events are being fired without proper timestamps
- The regime detector is evaluating all bars in a burst

**Root Cause Hypothesis**:
- The BacktestRunner in optimization mode might be replaying data too fast
- Event timestamps are being ignored or overwritten
- This explains why optimization sees "trending_up" while test sees "trending_down"
  (they're processing bars in different order/timing)

### 12. Event Ordering ❌ NOT VERIFIED
**Status**: Need to verify exact event sequence
- [ ] BAR event ordering
- [ ] CLASSIFICATION event ordering
- [ ] SIGNAL event ordering
- [ ] Event bus subscription order

**Next Steps**:
- Log all events with timestamps and sequence numbers
- Verify no race conditions

### 13. Additional Issues Found

#### Double Bar Counting Bug ✅ FIXED
- Regime detector was incrementing bar count twice
- Fixed in `on_bar()` method

#### Missing Regime Switching Enable ❌ NEEDS FIX
- Test run not enabling regime switching properly
- Need to fix detection of test mode

#### No Signal Generation ❌ CRITICAL
- Both runs producing 0 signals
- Likely due to:
  - Indicator warmup consuming too many bars
  - Rules canceling each other out
  - min_separation filter too restrictive
  - Rapid regime changes disrupting signals

## Priority Actions

1. **Fix Signal Generation** (HIGHEST)
   - Debug why no signals are generated
   - Check rule evaluation logic
   - Verify indicator readiness

2. **Fix Regime Detector Consistency**
   - Ensure same detector is used
   - Consistent warmup behavior
   - Fix bar counting

3. **Enable Regime Switching for Test**
   - Fix test mode detection
   - Ensure proper initialization

4. **Verify Event Flow**
   - Complete event ordering verification
   - Check for race conditions

## Root Cause Hypothesis

The main issues appear to be:
1. **Multiple regime detectors** causing confusion
2. **Inconsistent indicator warmup** between optimization and test
3. **No signal generation** due to parameter/warmup issues
4. **Rapid regime switching** (every 5-6 bars) disrupting strategy

## Recommended Solution

### Immediate Fix for Indicator Buffering:
1. **Ensure True Cold Start**: Modify indicator reset to:
   - Not just clear() the buffer
   - Reinitialize the buffer as a new deque
   - Reset all internal state variables
   - Force _ready = False

2. **Alternative: Consistent Warm Start**:
   - Save indicator state after training
   - Restore exact state for test phase
   - Ensure both runs start with same buffered data

### Long-term: Implement **scoped containerization** to ensure:
- Each run has its own isolated components
- No cross-contamination between detectors
- Consistent initialization and warmup
- Predictable event routing
- Separate indicator instances for train/test phases