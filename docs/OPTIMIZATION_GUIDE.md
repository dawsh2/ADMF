# Optimization Guide for Ensemble Strategy

This document explains how to optimize the ensemble trading strategy with multiple rules.

## Optimization Philosophy

The optimization process follows these principles:

1. **Rule Isolation**: Each rule is optimized independently to find its optimal parameters.
2. **Weight Optimization**: After individual rule optimization, genetic algorithm determines the optimal weights.
3. **Regime-Specific Optimization**: Parameters and weights are optimized for each market regime.

## Optimization Process

The optimization occurs in these phases:

### 1. Grid Search Optimization

First, rule parameters are optimized using grid search. You can optimize:

- All parameters at once (slowest)
- MA rule parameters only (faster)
- RSI rule parameters only (faster)

During grid search optimization, equal weights (0.5/0.5) are used for both rules.

### 2. Genetic Algorithm Optimization

After grid search, a genetic algorithm optimizes the weights between rules:

- Takes the best rule parameters from grid search
- Finds the optimal balance between rules (MA vs RSI)
- Ensures weights sum to 1.0

### 3. Adaptive Testing

Finally, adaptive tests are performed using:
- Regime-specific optimal parameters
- Weights optimized by genetic algorithm

## Command Line Options

```bash
# Optimize all parameters
python main.py --optimize --genetic-optimize --bars 1000

# Optimize only MA parameters
python main.py --optimize-ma --genetic-optimize --bars 1000

# Optimize only RSI parameters
python main.py --optimize-rsi --genetic-optimize --bars 1000

# Skip genetic optimization (use equal weights)
python main.py --optimize --bars 1000
```

## Optimization Tips

1. **Rule-Specific Optimization**: Use `--optimize-ma` or `--optimize-rsi` to run faster partial optimizations.
2. **Sequential Optimization**: First optimize MA, then RSI, then run genetic optimization for best results.
3. **Smaller Datasets**: Use `--bars` to limit the dataset for faster development iterations.
4. **Genetic Algorithm Settings**: Configure genetic algorithm in `config/config.yaml` to adjust population size, generations, etc.

## Performance Considerations

- Full optimization is computationally expensive - limit parameter ranges for initial testing
- Rule-specific optimization is much faster (potentially 4-8x)
- Genetic optimization is relatively fast compared to grid search
- Consider saving optimization results for later use/comparison

## Weight Interpretation

- Weights determine how strongly each rule "votes" for its signals
- Higher weight = stronger influence on decision making
- Optimal weights often vary by market regime
- Keep minimum weights above 0.3 to ensure rules can still generate signals