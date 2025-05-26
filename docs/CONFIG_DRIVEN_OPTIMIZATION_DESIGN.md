# Config-Driven Optimization Design

## Overview

This document outlines the design for a config-driven optimization system that allows users to define complex optimization workflows in their configuration files. The system supports multi-step optimization processes, including component-level optimization followed by ensemble weight optimization.

## Core Concepts

### 1. Optimization Workflow
A sequence of optimization steps defined in configuration, where each step can:
- Target specific components or groups
- Use different optimization methods
- Build upon results from previous steps
- Apply different objectives and constraints

### 2. Optimization Step Types

#### Component Optimization
Optimizes individual components (indicators, rules) independently:
```yaml
- step: component_optimization
  targets:
    - rsi_indicator
    - ma_crossover_rule
  method: grid_search
  objective: sharpe_ratio
  save_signals: true  # Save component signals for later use
```

#### Ensemble Weight Optimization
Optimizes weights for combining multiple components:
```yaml
- step: ensemble_weights
  method: genetic
  use_signals_from: component_optimization
  objective: regime_adaptive_performance
  constraints:
    - sum_to_one: true
    - non_negative: true
```

#### Regime-Specific Optimization
Optimizes parameters for each market regime:
```yaml
- step: regime_optimization
  base_results: component_optimization
  regimes: [trending_up, trending_down, ranging]
  method: grid_search
  objective: regime_sharpe_ratio
```

## Configuration Structure

### Basic Example - Rule-wise Optimization
```yaml
optimization:
  enabled: true
  output_dir: "optimization_results"
  
  workflow:
    # Step 1: Optimize each rule component
    - name: "optimize_rules"
      type: component_optimization
      targets:
        - component: rsi_oversold_rule
          parameter_space:
            rsi_period: [9, 14, 21]
            oversold_threshold: [20, 25, 30]
        - component: rsi_overbought_rule
          parameter_space:
            rsi_period: [9, 14, 21]
            overbought_threshold: [70, 75, 80]
        - component: ma_crossover_rule
          parameter_space:
            fast_period: [10, 20, 50]
            slow_period: [50, 100, 200]
      method: grid_search
      objective: 
        metric: sharpe_ratio
        minimize: false
      constraints:
        min_trades: 10
      evaluation:
        dataset: train
        
    # Step 2: Test optimized rules on validation set
    - name: "validate_rules"
      type: validation
      use_parameters_from: optimize_rules
      evaluation:
        dataset: test
        save_metrics: true
```

### Advanced Example - Ensemble with Signal Weights
```yaml
optimization:
  workflow:
    # Step 1: Component optimization
    - name: "optimize_components"
      type: component_optimization
      targets:
        - rsi_indicator
        - bollinger_bands
        - macd_indicator
      method: grid_search
      save_signals: true
      
    # Step 2: Optimize signal combination weights
    - name: "optimize_signal_weights"
      type: signal_weight_optimization
      signals_from: optimize_components
      method: genetic_algorithm
      parameters:
        population_size: 100
        generations: 50
        mutation_rate: 0.1
      objective:
        metric: risk_adjusted_return
        regimes: [trending_up, trending_down, ranging]
      constraints:
        - type: sum_to_one
        - type: non_negative
        - type: max_weight
          value: 0.5
          
    # Step 3: Regime-specific fine-tuning
    - name: "regime_tuning"
      type: regime_optimization
      base_parameters: optimize_signal_weights
      regimes: 
        trending_up:
          bias: momentum
          weight_adjustment: 1.2
        trending_down:
          bias: mean_reversion
          weight_adjustment: 0.8
        ranging:
          bias: neutral
          weight_adjustment: 1.0
```

## Implementation Architecture

### 1. Workflow Orchestrator
```python
class OptimizationWorkflowOrchestrator:
    """Executes optimization workflows defined in config."""
    
    def execute_workflow(self, workflow_config: List[Dict]) -> Dict[str, Any]:
        """Execute all steps in the workflow."""
        results = {}
        context = OptimizationContext()
        
        for step_config in workflow_config:
            step = self._create_step(step_config)
            step_results = step.execute(context)
            
            # Store results
            results[step_config['name']] = step_results
            
            # Update context for next steps
            context.add_results(step_config['name'], step_results)
            
        return results
```

### 2. Optimization Steps
```python
class OptimizationStep(ABC):
    """Base class for optimization workflow steps."""
    
    @abstractmethod
    def execute(self, context: OptimizationContext) -> Dict[str, Any]:
        """Execute the optimization step."""
        pass

class ComponentOptimizationStep(OptimizationStep):
    """Optimizes individual components."""
    
    def execute(self, context: OptimizationContext) -> Dict[str, Any]:
        results = {}
        
        for target in self.targets:
            component = context.get_component(target['component'])
            optimizer = ComponentOptimizer()
            
            # Run optimization
            opt_results = optimizer.optimize_component(
                component=component,
                method=self.method,
                objective_metric=self.objective['metric'],
                parameter_space=target.get('parameter_space')
            )
            
            results[target['component']] = opt_results
            
            # Save signals if requested
            if self.save_signals:
                context.save_component_signals(
                    target['component'], 
                    self._extract_signals(component, opt_results)
                )
                
        return results

class SignalWeightOptimizationStep(OptimizationStep):
    """Optimizes weights for combining signals."""
    
    def execute(self, context: OptimizationContext) -> Dict[str, Any]:
        # Get signals from previous step
        signals = context.get_signals(self.signals_from)
        
        # Create weight optimizer
        optimizer = SignalWeightOptimizer(
            signals=signals,
            method=self.method,
            objective=self.objective,
            constraints=self.constraints
        )
        
        # Run optimization
        return optimizer.optimize()
```

### 3. Integration with AppRunner
```python
class AppRunner(ComponentBase):
    def _run_optimization(self) -> Dict[str, Any]:
        """Run config-driven optimization."""
        
        # Get optimization workflow from config
        workflow_config = self.config.get('optimization.workflow', [])
        
        if not workflow_config:
            # Fall back to legacy optimization
            return self._run_legacy_optimization()
            
        # Create and run workflow orchestrator
        orchestrator = OptimizationWorkflowOrchestrator(
            container=self.container,
            config=self.config
        )
        
        results = orchestrator.execute_workflow(workflow_config)
        
        # Save results
        self._save_optimization_results(results)
        
        return results
```

## Benefits

1. **Flexibility**: Users can define arbitrarily complex optimization workflows
2. **Modularity**: Each step is independent and reusable
3. **Extensibility**: New step types can be added without changing core logic
4. **Traceability**: Each step's results are saved and can be analyzed
5. **Power**: Supports sophisticated strategies like signal weighting and regime adaptation

## Migration Path

1. Keep existing `--optimize` behavior as default
2. If `optimization.workflow` exists in config, use new system
3. Provide migration tool to convert old configs to new format
4. Eventually deprecate old optimization approach

## Next Steps

1. Implement OptimizationWorkflowOrchestrator
2. Create standard optimization step types
3. Build signal extraction and storage system
4. Add signal weight optimizer
5. Create examples and documentation