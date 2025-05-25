# Configuration Usage Examples

This document shows how to use the new Bootstrap-based ADMF system with different configurations.

## Basic Usage

### 1. Running a Simple Backtest

```bash
# Using the new minimal main.py
python main_ultimate.py --config config/example_backtest_config.yaml

# Optional: Override max bars
python main_ultimate.py --config config/example_backtest_config.yaml --bars 1000

# Optional: Enable debug logging
python main_ultimate.py --config config/example_backtest_config.yaml --log-level DEBUG --debug-log debug.log
```

The key insight: The config file determines that this runs in backtest mode, not command line args!

### 2. Running Optimization

```bash
# Just use a different config file!
python main_ultimate.py --config config/example_optimization_config.yaml

# The genetic optimization flags are now just hints to AppRunner
python main_ultimate.py --config config/example_optimization_config.yaml --genetic-optimize

# Random search
python main_ultimate.py --config config/example_optimization_config.yaml --random-search
```

### 3. Using Scoped Containers for Clean Optimization

```bash
# Use the scoped optimization config
python main_ultimate.py --config config/example_scoped_optimization_config.yaml
```

## Configuration Structure

### Determining Run Mode

The run mode is determined by `system.application_mode` in the config:

```yaml
system:
  application_mode: "backtest"  # or "optimization", "production", "test"
```

NOT by command line arguments!

### Entrypoint Configuration

Each run mode can have its own entrypoint component:

```yaml
system:
  run_modes:
    backtest:
      entrypoint_component: "app_runner"
    optimization:
      entrypoint_component: "app_runner"  
    production:
      entrypoint_component: "production_runner"  # Could be different!
```

### Component Configuration

Components are configured in the `components` section:

```yaml
components:
  data_handler_csv:
    csv_file_path: "data/SPY_1min.csv"
    max_bars: null  # Can be overridden by CLI
    
  strategy:
    # Strategy-specific config
```

## Advanced Features

### 1. Scoped Container Optimization

Enable scoped containers for complete isolation between optimization trials:

```yaml
system:
  optimization_settings:
    use_scoped_containers: true
    
components:
  optimizer:
    execution_mode: "scoped"
    scoped_components: ["portfolio_manager", "strategy"]
    shared_components: ["logger", "config"]
```

### 2. Dynamic Component Discovery

Place `component_meta.yaml` files in your source tree:

```yaml
# src/strategy/custom/component_meta.yaml
components:
  custom_strategy:
    class: "CustomStrategy"
    module: "strategy.custom.custom_strategy"
    dependencies: ["event_bus", "data_handler"]
    config_key: "components.custom_strategy"
```

Then the component is automatically available!

### 3. Parallel Optimization (Future)

With scoped containers, parallel trials become possible:

```yaml
components:
  genetic_optimizer:
    parallel_evaluation: true
    max_workers: 8
    use_scoped_contexts: true
```

## Key Principles

1. **Config Drives Behavior**: The configuration file determines what runs, not CLI args
2. **Components Are Managed**: Bootstrap handles all lifecycle management
3. **Clean Isolation**: Scoped containers prevent state pollution
4. **Extensibility**: Add new components without changing code

## Migration from Old System

### Old Way:
```bash
python main.py --optimize --genetic-optimize --bars 1000
```

### New Way:
```bash
# Create optimization config with application_mode: "optimization"
python main_ultimate.py --config config/my_optimization.yaml --genetic-optimize --bars 1000
```

The CLI args are now just hints/overrides, not mode determinants!