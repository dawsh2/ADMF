# Ultimate Minimal main.py Architecture

## Overview

This architecture achieves the goal of an ultra-minimal main.py (only 22 lines!) by properly separating concerns across three layers:

```
main.py (22 lines)
   ↓ passes sys.argv
ApplicationLauncher
   ↓ parses args, loads config, sets up Bootstrap
Bootstrap
   ↓ creates and manages all components
AppRunner (Component)
   ↓ executes application logic
```

## File Breakdown

### 1. main_ultimate.py (22 lines)
```python
def main():
    launcher = ApplicationLauncher(sys.argv[1:])
    sys.exit(launcher.run())
```
- **Only responsibility**: Capture and forward command line arguments
- No imports except sys and ApplicationLauncher
- No error handling (delegated)
- No logging setup (delegated)
- No config loading (delegated)

### 2. ApplicationLauncher (~150 lines)
**Responsibilities:**
- Parse command line arguments
- Set up bootstrap logging
- Load configuration
- Determine run mode from config (not args!)
- Configure Bootstrap with AppRunner as entrypoint
- Handle top-level exceptions

**Key insight**: Run mode comes from config, not command line args

### 3. AppRunner (ComponentBase) (~200 lines)
**Responsibilities:**
- Receive CLI args via context.metadata
- Execute appropriate logic based on run mode
- Orchestrate other components (optimizer, portfolio, etc.)
- Return results

**Benefits of being a Component:**
- Proper lifecycle management (initialize, start, stop, teardown)
- Dependency injection via context
- Testable in isolation
- Follows established patterns

## Configuration-Driven Behavior

The config file determines what runs:

```yaml
system:
  application_mode: "optimization"  # or "backtest", "production", "test"
  run_modes:
    optimization:
      entrypoint_component: "app_runner"
    backtest:
      entrypoint_component: "app_runner"
```

## Command Line Arguments Flow

```
User types: python main.py --config prod.yaml --bars 1000 --genetic-optimize

main.py:
  sys.argv = ['main.py', '--config', 'prod.yaml', '--bars', '1000', '--genetic-optimize']
  → ApplicationLauncher(sys.argv[1:])

ApplicationLauncher:
  Parses: {'config': 'prod.yaml', 'bars': 1000, 'genetic_optimize': True}
  → Passes via context.metadata['cli_args']

AppRunner:
  Receives in _initialize(): self.context.metadata['cli_args']
  Uses: self.max_bars = 1000, self.optimize_flags['genetic_optimize'] = True
```

## Benefits

1. **Minimal main.py**: Just 22 lines, truly just an entry point
2. **Clear separation**: Each layer has one responsibility
3. **Config-driven**: Application behavior determined by config, not CLI
4. **Testable**: Each component can be tested independently
5. **Extensible**: Easy to add new run modes or modify behavior
6. **Standard patterns**: AppRunner follows ComponentBase lifecycle

## Comparison to Original

| Aspect | Original main.py | Ultimate Architecture |
|--------|-----------------|----------------------|
| Lines in main.py | 511 | 22 |
| Responsibilities | 10+ | 1 |
| Config determines behavior | No | Yes |
| Component lifecycle | Manual | Automatic |
| Testability | Difficult | Easy |
| Adding new modes | Modify main.py | Config only |

## Migration Path

1. Keep existing main.py
2. Add main_ultimate.py as alternative entry point
3. Test with: `python main_ultimate.py --config config/config.yaml`
4. Gradually migrate functionality
5. Eventually replace main.py

This architecture achieves the vision of main.py being a pure entry point with all logic properly encapsulated in testable, manageable components.