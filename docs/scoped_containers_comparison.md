# Scoped Containers: Complexity Comparison

## Current Approach (Shared Container + Reset)

```python
# COMPLEX: Must carefully reset state between trials
def run_backtest(params):
    # Get shared components
    portfolio = container.resolve('portfolio')
    strategy = container.resolve('strategy')
    data_handler = container.resolve('data_handler')
    
    # Complex reset logic needed
    portfolio.reset()  # Hope this clears everything!
    strategy.reset()   # Hope parameters don't leak!
    data_handler.reset_to_start()  # Hope data position resets!
    
    # Clear event subscriptions (error-prone)
    event_bus.unsubscribe_all(portfolio)
    event_bus.unsubscribe_all(strategy)
    # Re-subscribe (hope we don't miss any)
    portfolio.initialize_event_subscriptions()
    strategy.initialize_event_subscriptions()
    
    # Set new parameters
    strategy.set_parameters(params)
    
    # Run backtest
    # ... 
    
    # State pollution risk: Did we reset everything?
    # Hidden dependencies might retain state
```

### Problems:
- 🔴 Complex reset logic in every component
- 🔴 Easy to miss resetting some state
- 🔴 Event subscriptions can leak
- 🔴 Parameters might not fully update
- 🔴 Hidden state in third-party libraries
- 🔴 Race conditions in shared components

## New Approach (Scoped Containers)

```python
# SIMPLE: Fresh components for each trial
def run_backtest(params):
    # Create isolated scope
    scoped_context = bootstrap.create_scoped_context(f"trial_{id}")
    
    # Create fresh components - no state from previous runs!
    portfolio = BasicPortfolio(scoped_context.event_bus)
    strategy = MAStrategy(scoped_context.event_bus)
    
    # Register in scoped container
    scoped_context.container.register('portfolio', portfolio)
    scoped_context.container.register('strategy', strategy)
    
    # Set parameters on fresh instance
    strategy.set_parameters(params)
    
    # Initialize (fresh event subscriptions)
    portfolio.initialize(scoped_context)
    strategy.initialize(scoped_context)
    
    # Run backtest
    # ...
    
    # Cleanup is automatic when scope ends
    # No state pollution possible!
```

### Benefits:
- ✅ No complex reset logic needed
- ✅ Guaranteed fresh state
- ✅ Event isolation built-in
- ✅ Parameters always clean
- ✅ No hidden state issues
- ✅ Thread-safe by design

## Implementation Complexity

### What We Added:
1. **Container.py**: ~15 lines (parent support)
2. **Bootstrap.py**: ~50 lines (create_scoped_context method)

### What We Can Remove:
1. **Complex reset() methods** in every component
2. **State validation logic**
3. **Event subscription cleanup code**
4. **Parameter change detection**
5. **State pollution debugging**

## Net Result: SIMPLER

The scoped container approach is actually **less complex** overall because:

1. **Component Design**: Components don't need reset logic
2. **Debugging**: No state pollution mysteries
3. **Testing**: Each test gets clean components
4. **Parallelization**: Ready for parallel execution
5. **Maintenance**: Fewer edge cases to handle

## Migration Path

### Phase 1: Add Scoped Support (DONE ✅)
- Extended Container with parent support
- Added create_scoped_context to Bootstrap

### Phase 2: Update Optimizers (NEXT)
- Modify optimizers to use scoped contexts
- Remove complex reset logic

### Phase 3: Simplify Components
- Remove reset() methods
- Remove state validation
- Components become simpler

### Phase 4: Enable Parallelization
- With isolation, can run trials in parallel
- Massive performance improvement possible

## Conclusion

The scoped container approach is:
- **Conceptually cleaner**: Isolation by design
- **Easier to implement**: Less code overall  
- **More reliable**: Eliminates entire categories of bugs
- **Future-proof**: Enables parallelization

The "complexity" is just proper separation of concerns, which actually makes the system simpler!