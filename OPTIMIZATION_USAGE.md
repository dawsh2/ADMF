# Optimization Usage Guide

## Current Status

The new `main_ultimate.py` with ComponentBase architecture does NOT support optimization mode yet.

## How to Run Optimization

Use the original `main.py` for optimization tasks:

```bash
# Basic optimization
python main.py --config config/config.yaml --optimize --bars 100

# Optimize specific parameters
python main.py --config config/config.yaml --optimize-ma --bars 100
python main.py --config config/config.yaml --optimize-rsi --bars 100

# Genetic optimization
python main.py --config config/config.yaml --genetic-optimize --bars 100
```

## Why Optimization Doesn't Work in New Architecture

The new ComponentBase architecture needs significant changes to support optimization:

1. **Component Reset** - Components need to be reset between optimization iterations
2. **Parameter Injection** - Need a way to inject different parameters into strategies
3. **Isolation** - Each optimization iteration needs isolated component instances
4. **Data Streaming** - The data handler needs to re-stream data for each iteration
5. **Result Collection** - Need to collect and compare results across iterations

## Future Implementation Plan

To properly implement optimization in the new architecture:

1. Create an `OptimizationEntrypoint` component similar to `BacktestRunner`
2. Use scoped contexts to isolate each optimization iteration
3. Implement parameter injection system for strategies
4. Add component reset functionality
5. Create result aggregation and comparison system

## Files Created During Failed Attempt

- `optimization_results_TIMESTAMP.json` - Contains empty results from failed attempts
- `best_parameters.json` - Contains parameters from failed optimization

These can be safely deleted.