# main.py Refactoring Comparison

## Original main.py Issues

The original `main.py` had **511 lines** with many responsibilities:

1. **Manual Component Registration** (lines 112-229)
   - Hardcoded registration of every component
   - Complex constructor arguments
   - Repetitive boilerplate

2. **Complex Optimization Logic** (lines 208-359)
   - 150+ lines of optimization-specific code
   - Nested if/else chains for different optimization modes
   - Manual component lifecycle management

3. **Duplicate Logic** (lines 378-503)
   - Separate `run_application_logic()` function
   - Duplicated component setup/start/stop logic
   - Manual event flow coordination

4. **State Management Issues**
   - Manual component ordering to avoid race conditions
   - Complex dependency resolution
   - Error-prone cleanup sequences

## Refactored main.py Benefits

The new `main_refactored.py` has **~250 lines** with clear separation:

### 1. Clean Argument Parsing
```python
# Old: 30+ arguments mixed together
parser.add_argument("--optimize", action="store_true", help="...")
parser.add_argument("--optimize-ma", action="store_true", help="...")
parser.add_argument("--optimize-rsi", action="store_true", help="...")
# ... many more

# New: Simple, focused arguments
parser.add_argument("--mode", choices=['production', 'backtest', 'optimization', 'test'])
parser.add_argument("--config", default="config/config.yaml")
parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
```

### 2. Bootstrap Handles Everything
```python
# Old: 100+ lines of manual component setup
container.register_type("data_handler", CSVDataHandler, True, constructor_kwargs=csv_args)
container.register_type("strategy", EnsembleSignalStrategy, True, constructor_kwargs=ensemble_strat_args)
# ... many more registrations

# New: One line!
components = bootstrap.setup_managed_components()
```

### 3. Config-Driven Execution
```python
# Old: Hardcoded logic for each mode
if run_optimization_mode:
    # 150+ lines of optimization logic
elif run_genetic_optimization:
    # More specific logic
else:
    # Standard backtest logic

# New: Config determines what runs
result = bootstrap.execute_entrypoint()  # Config decides which component
```

### 4. Automatic Lifecycle Management
```python
# Old: Manual component lifecycle
for comp in components_to_manage:
    comp.setup()
    if comp.get_state() == BaseComponent.STATE_INITIALIZED:
        comp.start()
    # Error handling...

# New: Bootstrap handles it all
bootstrap.start_components()  # Starts in dependency order
```

### 5. Clean Resource Management
```python
# Old: Complex finally block with manual cleanup
finally:
    for comp in reversed(components_to_manage):
        if comp and isinstance(comp, BaseComponent):
            if comp.get_state() not in [...]:
                comp.stop()
    # More cleanup...

# New: Context manager handles everything
with Bootstrap() as bootstrap:
    # ... run application
# Automatic cleanup on exit
```

## Key Improvements

1. **Separation of Concerns**
   - main.py only handles CLI and orchestration
   - Bootstrap manages components
   - Config drives behavior

2. **Flexibility**
   - Add new components without changing main.py
   - Change entrypoints via config
   - Support new run modes easily

3. **Maintainability**
   - 50% less code
   - Clear, linear flow
   - No duplicate logic

4. **Reliability**
   - Automatic dependency ordering
   - Proper cleanup guaranteed
   - No manual state management

5. **Extensibility**
   - Easy to add new run modes
   - Plugin-style component addition
   - Config-driven customization

## Migration Path

The refactored version includes backward compatibility:
- Legacy --optimize flag maps to --mode optimization
- `run_legacy_logic()` supports components without execute()
- Gradual migration possible

## Summary

The refactoring transforms main.py from a complex, monolithic entry point into a clean orchestrator that:
- Reduces code by 50%
- Eliminates manual component management
- Provides better error handling
- Enables configuration-driven behavior
- Maintains backward compatibility