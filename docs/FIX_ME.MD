# FIX_ME.MD - Issues to Address

## 0. Container Access for Dynamic Component Creation

### Current Issue
The workflow orchestrator and component optimizer need to create components dynamically at runtime (e.g., creating evaluators during optimization). Currently, we pass the container or required components through multiple layers, which feels like it might defeat the purpose of dependency injection.

### Current Solution
Components access the container through `self._context.container` and pass required components as parameters:

```python
# In workflow orchestrator
component_optimizer.set_isolated_evaluator(
    backtest_runner=backtest_runner,
    data_handler=data_handler,
    portfolio=portfolio_manager,
    risk_manager=risk_manager,
    execution_handler=execution_handler
)
```

### Better Solution - Factory Pattern
Register component factories in the container that capture dependencies at registration time:

```python
# During bootstrap/initialization
def create_optimizer_factory(backtest_runner, data_handler, portfolio, risk_manager, execution_handler):
    """Factory that creates optimizers with all dependencies pre-configured."""
    def factory():
        optimizer = ComponentOptimizer(instance_name=f"optimizer_{uuid.uuid4().hex[:8]}")
        optimizer.set_isolated_evaluator(
            backtest_runner=backtest_runner,
            data_handler=data_handler,
            portfolio=portfolio,
            risk_manager=risk_manager,
            execution_handler=execution_handler
        )
        return optimizer
    return factory

# Register the factory
container.register_factory(
    'component_optimizer_factory',
    create_optimizer_factory(
        container.resolve('backtest_runner'),
        container.resolve('data_handler'),
        container.resolve('portfolio_manager'),
        container.resolve('risk_manager'),
        container.resolve('execution_handler')
    )
)

# Then in workflow orchestrator
optimizer = self._context.container.create('component_optimizer_factory')
# optimizer comes pre-configured with all dependencies
```

### Benefits
1. **Maintains DI principles** - Dependencies are injected at registration time
2. **No container passing** - Components don't need direct container access
3. **Cleaner code** - No parameter drilling through multiple layers
4. **Type safety** - Factory signature clearly shows what's needed
5. **Testability** - Easy to register test factories with mocks

### Implementation Notes
- Would need to add `register_factory()` and `create()` methods to Container
- Factories should be registered during bootstrap after core components are available
- Consider using a naming convention like `*_factory` for all factories

### Example Container Enhancement
```python
class Container:
    def __init__(self):
        self._instances = {}
        self._providers = {}
        self._factories = {}  # New: store factory functions
    
    def register_factory(self, name: str, factory_func: Callable[[], Any]):
        """Register a factory function that creates new instances."""
        self._factories[name] = factory_func
        
    def create(self, factory_name: str) -> Any:
        """Create a new instance using a registered factory."""
        if factory_name not in self._factories:
            raise DependencyNotFoundError(f"Factory '{factory_name}' not registered")
        return self._factories[factory_name]()
```

This pattern is especially useful for components that need to be created dynamically during runtime, like optimizers, evaluators, and temporary strategies for isolated testing.

## 1. Regime Analyzer Trade Count Issue (FIXED)

The regime statistics were showing "Total trades: 0" even though trades were being tracked by regime. This was because the regime analyzer wasn't storing the trade counts from each parameter combination.

**Fix Applied**: Added `trade_counts` list to store trade counts from each parameter combination and updated the statistics calculation to sum these counts.

## 2. Total Bars Checked Misleading

The regime detector summary shows cumulative bars across all optimization iterations:
```
Total bars checked: 40000  # With 4 parameter combinations and 10000 bars each
```

This should be divided by the number of unique parameter combinations to show the actual bars per backtest.

## 3. MyPrimaryRegimeDetector Shows No Checks

At teardown, there's a second regime detector (`MyPrimaryRegimeDetector`) that shows:
```
=== Regime Detection Summary ===
No regime detection checks were performed
=== End of Summary ===
```

This appears to be a duplicate/unused regime detector that should be removed from the configuration.

## 4. Regime-Specific Parameter Selection

With 10,000 bars, we now see some differentiation in parameter selection:
- **Default regime**: Selected fast=10, slow=30 (different from others!)
- **Other regimes**: All selected fast=5, slow=20

This shows the optimization is working, but the limited parameter space (only 4 combinations) may not provide enough diversity for truly regime-specific optimization.

## 5. Lack of Scoped Containers in Optimization

### Current Issue
The optimization framework reuses the same container and event bus across all optimization trials, leading to:
- State leakage between optimization runs
- Isolated strategy instances persisting and subscribing to events after their evaluation
- Duplicate output from multiple isolated strategies processing the same events
- Potential interference between parameter combinations

### Symptoms
During test phase after optimization, we see multiple active subscribers:
```
Active subscribers to CLASSIFICATION: 15
- 9 instances of IsolatedStrategy 'isolated_strategy_ma_crossover'
- 4 instances of IsolatedStrategy 'isolated_strategy_rsi'
- 1 instance of RegimeAdaptiveEnsembleComposed 'strategy'
```

### Root Cause
As documented in `docs/modules/core/2_CORE_BOOTSTRAP_CONFIGURATION_DEPENDENCIES.MD`, each independent execution unit (like optimization trials) should have its own scoped container for proper state isolation. Currently, the optimization engine reuses the global container.

### Proper Solution - Scoped Containers
Each optimization trial should:
1. Create a new, fresh Container instance
2. Register all required components in this scoped container
3. Run the backtest with complete isolation
4. Dispose of the entire container after evaluation

### Current Workaround
Added cleanup in `isolated_evaluator.py` to explicitly:
- Unsubscribe isolated strategies from all events
- Stop and teardown isolated strategies
- Clear container registrations

This helps but doesn't address the fundamental architectural issue.

### Implementation Requirements
1. Modify optimization engine to create scoped containers per trial
2. Ensure event bus isolation between trials
3. Implement proper container lifecycle management
4. Follow the pattern shown in conceptual `CleanBacktestEngine`

## Next Steps

1. Implement proper trade counting in regime analyzer (or accept current limitation)
2. Fix the bars checked display to show per-backtest count
3. Remove or properly configure MyPrimaryRegimeDetector
4. Consider expanding parameter space for better regime differentiation
5. Implement scoped containers for optimization trials to ensure proper isolation