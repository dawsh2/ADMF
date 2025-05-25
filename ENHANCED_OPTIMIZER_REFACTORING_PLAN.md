# EnhancedOptimizer Refactoring Plan

## Current State Analysis
- **File Size**: 2,224 lines (too large for single responsibility)
- **Main Class**: `EnhancedOptimizer` extends `BasicOptimizer`

## Identified Responsibilities (based on method analysis)

### 1. Backtest Execution
- `_perform_single_backtest_run()` (line 49)
- Handles single backtest iteration setup and execution
- **Target Module**: `backtest_engine.py`

### 2. Optimization Methods
- `run_grid_search()` (line 252) - Grid search implementation
- `run_per_regime_genetic_optimization()` (line 511) - Genetic algorithm orchestration
- `run_per_regime_random_search_optimization()` (line 619) - Random search
- **Target Module**: `optimization_methods/` directory with:
  - `grid_search.py`
  - `genetic_method.py`
  - `random_search.py`

### 3. Regime-Specific Processing
- `_process_regime_performance()` (line 117)
- Regime-specific parameter tracking and analysis
- **Target Module**: `regime_processor.py`

### 4. Adaptive Testing
- `run_adaptive_test()` (line 707)
- `_setup_adaptive_mode()` (line 1022)
- `_run_regime_adaptive_test()` (line 1103)
- **Target Module**: `adaptive_test_manager.py`

### 5. Results Management
- `_log_optimization_results()` (line 784)
- `_save_results_to_file()` (line 1901)
- `_get_top_n_performers()` (line 751)
- **Target Module**: `results_manager.py`

### 6. Rule-wise Optimization
- `_run_rulewise_optimization()` (line 1974)
- `_combine_rulewise_parameters()` (line 2148)
- **Target Module**: `rulewise_optimizer.py`

## Proposed New Structure

```
src/strategy/optimization/
├── enhanced_optimizer.py (reduced to orchestrator role)
├── core/
│   ├── __init__.py
│   ├── backtest_engine.py
│   ├── optimization_manager.py (new main orchestrator)
│   └── optimization_target.py
├── methods/
│   ├── __init__.py
│   ├── base_method.py
│   ├── grid_search.py
│   ├── genetic_method.py
│   └── random_search.py
├── sequences/
│   ├── __init__.py
│   ├── base_sequence.py
│   ├── regime_specific_sequence.py
│   ├── rulewise_sequence.py
│   └── adaptive_test_sequence.py
├── processors/
│   ├── __init__.py
│   ├── regime_processor.py
│   └── parameter_processor.py
├── results/
│   ├── __init__.py
│   ├── results_manager.py
│   └── results_analyzer.py
└── testing/
    ├── __init__.py
    └── adaptive_test_manager.py
```

## Implementation Strategy

### Phase 1: Create Module Structure
1. Create directory structure
2. Define base classes and interfaces
3. Create empty implementations

### Phase 2: Extract Backtest Engine (High Priority)
1. Move `_perform_single_backtest_run()` to `BacktestEngine`
2. Define clear interface for backtest execution
3. Update EnhancedOptimizer to use BacktestEngine

### Phase 3: Extract Results Management
1. Move all results-related methods to `ResultsManager`
2. Create standardized results interfaces
3. Simplify EnhancedOptimizer's result handling

### Phase 4: Extract Optimization Methods
1. Create base `OptimizationMethod` class
2. Move grid search logic to `GridSearchMethod`
3. Create adapters for genetic and random search

### Phase 5: Extract Adaptive Testing
1. Move adaptive test logic to `AdaptiveTestManager`
2. Ensure clean separation from optimization logic
3. Align with production run behavior

### Phase 6: Create Optimization Manager
1. Implement new `OptimizationManager` as main orchestrator
2. Migrate orchestration logic from EnhancedOptimizer
3. Deprecate direct use of EnhancedOptimizer

## Key Principles
1. Each module should have a single, clear responsibility
2. Use dependency injection for component access
3. Define clear interfaces between modules
4. Maintain backward compatibility during transition
5. Write tests for each new module before moving code

## Next Steps
1. Review and approve this plan
2. Create the directory structure
3. Start with Phase 2 (BacktestEngine) as it's needed for Part 1 of the action plan
4. Incrementally refactor while maintaining functionality