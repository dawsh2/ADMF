# Config-Driven Optimization Workflow

## Overview

The config-driven optimization workflow allows you to define complex optimization sequences directly in your configuration file. When you run with `--optimize`, the system executes the workflow steps defined in your config, enabling sophisticated multi-stage optimization strategies.

## Architecture

### Key Components

1. **OptimizationWorkflowOrchestrator** - Orchestrates the execution of workflow steps
2. **ComponentOptimizer** - Handles component-based (rulewise) optimization
3. **EnhancedOptimizer** - Handles regime and ensemble optimization
4. **AppRunner** - Modified to use workflow orchestrator when available

### Workflow Execution

When `--optimize` is specified:
1. AppRunner checks for a `workflow_orchestrator` component
2. If found, it delegates to the workflow orchestrator
3. The orchestrator reads the workflow definition from config
4. Each step is executed in order, respecting dependencies
5. Results are saved and passed between steps

## Configuration

### Basic Structure

```yaml
optimization:
  # Date ranges for train/test split
  train_date_range: ["2023-01-01", "2023-06-30"]
  test_date_range: ["2023-07-01", "2023-12-31"]
  
  # Workflow definition
  workflow:
    - name: "step_name"
      type: "optimization_type"
      # Step-specific configuration
      depends_on: ["previous_step"]  # Optional
```

### Workflow Step Types

#### 1. Rulewise Optimization

Optimizes individual components (indicators, rules):

```yaml
- name: "component_optimization"
  type: "rulewise"
  targets:
    - "rsi_indicator_*"    # Pattern matching
    - "ma_indicator_*"     # Multiple patterns
  method: "grid_search"    # or "genetic"
  generations: 50         # For genetic method
  population_size: 100    # For genetic method
```

#### 2. Regime Optimization

Optimizes parameters per market regime:

```yaml
- name: "regime_optimization"
  type: "regime_optimization"
  regime_config:
    regimes_to_optimize: ["TRENDING_UP", "TRENDING_DOWN", "VOLATILE", "DEFAULT"]
    min_trades_per_regime: 10
  method: "grid_search"
```

#### 3. Ensemble Weight Optimization

Optimizes signal weights:

```yaml
- name: "ensemble_weight_optimization"
  type: "ensemble_weights"
  method: "genetic"
  generations: 50
  population_size: 100
  depends_on: ["component_optimization", "regime_optimization"]
```

## Example Workflows

### Simple Component Optimization

```yaml
optimization:
  workflow:
    - name: "optimize_indicators"
      type: "rulewise"
      targets: ["rsi_indicator_*"]
      method: "grid_search"
```

### Complex Multi-Stage Optimization

```yaml
optimization:
  workflow:
    # First optimize individual components
    - name: "component_optimization"
      type: "rulewise"
      targets: ["rsi_indicator_*", "ma_indicator_*"]
      method: "grid_search"
      
    # Then optimize regime parameters
    - name: "regime_optimization"
      type: "regime_optimization"
      method: "grid_search"
      depends_on: ["component_optimization"]
      
    # Finally optimize ensemble weights
    - name: "ensemble_weights"
      type: "ensemble_weights"
      method: "genetic"
      generations: 100
      depends_on: ["component_optimization", "regime_optimization"]
```

## Implementation Status

### Completed
- [x] OptimizationWorkflowOrchestrator implementation
- [x] Integration with AppRunner
- [x] Bootstrap component registration
- [x] Config-driven workflow execution
- [x] Dependency management between steps
- [x] Result persistence and tracking

### Pending
- [ ] Full implementation of component evaluation
- [ ] Integration with actual backtest runs
- [ ] Progress reporting and visualization
- [ ] Parallel execution of independent steps
- [ ] Checkpoint/resume functionality

## Usage

1. Create a config file with optimization workflow:
   ```yaml
   optimization:
     workflow:
       - name: "my_optimization"
         type: "rulewise"
         targets: ["*indicator*"]
         method: "grid_search"
   ```

2. Run with optimize flag:
   ```bash
   python main_ultimate.py --config config/my_workflow.yaml --optimize
   ```

3. Results are saved to `optimization_results/` directory

## Benefits

1. **Declarative**: Define optimization strategy in config, not code
2. **Flexible**: Mix and match different optimization types
3. **Reusable**: Save and share optimization workflows
4. **Traceable**: All results are persisted with workflow metadata
5. **Extensible**: Easy to add new optimization types

## Future Enhancements

1. **Parallel Execution**: Run independent steps concurrently
2. **Conditional Steps**: Execute steps based on previous results
3. **Custom Metrics**: Define optimization objectives in config
4. **Visualization**: Generate optimization reports and plots
5. **Cloud Integration**: Distribute optimization across multiple machines