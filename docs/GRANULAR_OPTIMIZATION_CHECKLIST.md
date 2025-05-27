# Granular Optimization Implementation Checklist

## Overview
This checklist tracks the implementation of granular optimization features as described in OPTIMIZATION.MD, focusing on rule isolation, weight optimization, and regime-specific parameter tuning.

## Current State
- ✅ Basic workflow orchestrator implemented
- ✅ Component optimizer working with existing infrastructure
- ✅ Config-driven workflow execution
- ✅ Rule isolation implemented (99% reduction in backtests!)
- ⏸️ Signal capture and replay for weight optimization (deferred)
- ❌ Regime-specific workflow stages (next priority)

## Phase 1: Rule Isolation Framework

### 1.1 Isolated Component Evaluator
- [x] Create `IsolatedComponentEvaluator` class
  - [x] Takes a single rule/indicator component
  - [x] Creates minimal strategy wrapper with just that component
  - [x] Runs backtest with single-component strategy
  - [x] Returns performance metrics
- [x] Integrate with `ComponentOptimizer`
  - [x] Add `isolate` parameter to optimization config
  - [x] Switch between full strategy vs isolated evaluation

### 1.2 Workflow Step Enhancement
- [x] Update `ComponentOptimizationStep` to support isolation
  - [x] Add `isolate: true` config option
  - [x] Create isolated evaluator when requested
  - [x] Store isolation metadata in results
- [x] Test with example configurations

## Phase 2: Signal Capture for Weight Optimization (DEFERRED)
*Note: Deferred since rule isolation already provides 99% reduction in backtests. This is a further optimization for later.*

### 2.1 Signal Collector
- [ ] Create `SignalCollector` class
  - [ ] Listens to `SIGNAL` events during backtest
  - [ ] Captures: timestamp, component name, signal strength, regime
  - [ ] Stores in efficient format (e.g., parquet/pickle)
- [ ] Integrate with optimization workflow
  - [ ] Add `capture_signals: true` option
  - [ ] Attach collector during parameter optimization
  - [ ] Save signals to designated directory

### 2.2 Signal Replayer
- [ ] Create `SignalReplayer` class
  - [ ] Loads captured signals from file
  - [ ] Replays signals with different weights
  - [ ] Calculates performance without full backtest
  - [ ] Handles regime filtering
- [ ] Create `SignalWeightOptimizer`
  - [ ] Uses replayer instead of full backtests
  - [ ] Supports various weight optimization methods
  - [ ] Enforces weight constraints (sum to 1, etc.)

### 2.3 Weight Optimization Step
- [ ] Create `WeightOptimizationStep` class
  - [ ] Loads signals from previous steps
  - [ ] Runs weight optimization per regime
  - [ ] Saves optimal weights with parameters
- [ ] Update workflow orchestrator to support this step type

## Testing Instructions

To test the isolated optimization:
```bash
python main_ultimate.py --config config/test_ensemble_optimization.yaml --bars 200 --optimize
```

This configuration includes:
- MA rule optimization in isolation
- RSI rule optimization in isolation  
- Weight optimization after finding best parameters
- Regime-specific optimization (pending implementation)

## Phase 3: Regime-Specific Optimization (NEXT PRIORITY)

### 3.1 Regime-Aware Evaluator
- [ ] Create `RegimeAwareEvaluator` class
  - [ ] Filters performance metrics by regime
  - [ ] Requires minimum trades per regime
  - [ ] Handles regime transition trades appropriately
- [ ] Update component optimizer for regime support
  - [ ] Add `regime` parameter to optimization
  - [ ] Use regime-filtered metrics for evaluation

### 3.2 Regime Optimization Workflow
- [ ] Create `RegimeOptimizationStep` class
  - [ ] Runs optimization for each regime separately
  - [ ] Uses regime-aware evaluator
  - [ ] Consolidates results into regime-keyed structure
- [ ] Support sequential regime optimization
  - [ ] Optimize parameters per regime
  - [ ] Then optimize weights per regime
  - [ ] Save complete regime-specific configuration

## Phase 4: Advanced Workflow Features

### 4.1 Dependency Management
- [ ] Implement `use_results_from` for workflow steps
  - [ ] Load parameters from previous step
  - [ ] Apply to components before optimization
  - [ ] Support parameter freezing
- [ ] Add validation for workflow dependencies

### 4.2 Constraint System
- [ ] Implement parameter constraints
  - [ ] Range constraints (min/max)
  - [ ] Relational constraints (param1 < param2)
  - [ ] Custom constraint functions
- [ ] Add weight constraints
  - [ ] Sum to one
  - [ ] Non-negative
  - [ ] Maximum weight limits

### 4.3 Progress and Monitoring
- [ ] Add progress reporting
  - [ ] Current step and progress within step
  - [ ] Estimated time remaining
  - [ ] Best parameters found so far
- [ ] Implement checkpoint/resume
  - [ ] Save workflow state periodically
  - [ ] Resume from last checkpoint
  - [ ] Handle partial results

## Phase 5: Configuration and Integration

### 5.1 Configuration Schema
- [ ] Define comprehensive workflow schema
  - [ ] Document all step types and options
  - [ ] Add validation for workflow configs
  - [ ] Create example configurations
- [ ] Migration guide from current to new system

### 5.2 Results Management
- [ ] Enhance results storage
  - [ ] Include workflow metadata
  - [ ] Track optimization lineage
  - [ ] Support result comparison/analysis
- [ ] Create results visualization tools

### 5.3 Testing and Documentation
- [ ] Unit tests for all new components
- [ ] Integration tests for complete workflows
- [ ] Performance benchmarks
- [ ] User documentation and tutorials

## Example Target Configuration

```yaml
optimization:
  workflow:
    # Step 1: Optimize MA rule in isolation
    - name: "optimize_ma_isolated"
      type: "component"
      config:
        target: "ma_crossover_rule"
        isolate: true
        method: "grid_search"
        capture_signals: true
        
    # Step 2: Optimize RSI rule in isolation  
    - name: "optimize_rsi_isolated"
      type: "component"
      config:
        target: "rsi_rule"
        isolate: true
        method: "grid_search"
        capture_signals: true
        
    # Step 3: Optimize weights using captured signals
    - name: "optimize_ensemble_weights"
      type: "weight_optimization"
      config:
        signals_from: ["optimize_ma_isolated", "optimize_rsi_isolated"]
        method: "genetic"
        per_regime: true
        
    # Step 4: Regime-specific fine-tuning
    - name: "regime_specific_tuning"
      type: "regime_optimization"
      config:
        base_parameters: "optimize_ensemble_weights"
        regimes: ["trending_up", "trending_down", "volatile", "default"]
        fine_tune_parameters: true
        fine_tune_weights: true
```

## Success Metrics
- [ ] Reduce optimization time by >90% through isolation
- [ ] Support complex multi-stage workflows
- [ ] Maintain or improve optimization quality
- [ ] Enable new optimization strategies (signal-based weights)
- [ ] Full backward compatibility with existing configs