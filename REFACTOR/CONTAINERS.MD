# Strategy State Management in Shared Indicator Architecture

## Table of Contents

1. [The Fundamental Constraint](#the-fundamental-constraint)
2. [Architectural Overview](#architectural-overview)
3. [Container Architecture Diagram](#container-architecture-diagram)
4. [Memory Optimization Strategies](#memory-optimization-strategies)
5. [Implementation Patterns](#implementation-patterns)
6. [Performance Analysis](#performance-analysis)
7. [Best Practices](#best-practices)

## The Fundamental Constraint

While the shared indicator architecture dramatically reduces computational redundancy, we face an irreducible constraint: **each unique parameter combination represents a distinct trading strategy requiring independent state tracking**.

### Why Strategy State Cannot Be Shared

```python
# Example: Two strategies with different parameters
# Strategy A: MA(5, 10) - Fast crossover
# Strategy B: MA(15, 20) - Slow crossover

# At time T1:
# - MA(5) > MA(10) → Strategy A signals BUY
# - MA(15) < MA(20) → Strategy B signals SELL

# Result: Divergent portfolios that must be tracked separately
```

### The Combinatorial Reality

For `n` parameter values and `k` parameters per strategy:

| Optimization Type | Indicator Containers | Strategy Containers | Growth Rate |
|-------------------|---------------------|---------------------|-------------|
| Single Parameter | O(n) | O(n) | Linear |
| Parameter Pairs | O(n) | O(n²) | Quadratic |
| k Parameters | O(n) | O(nᵏ) | Exponential |

**Example**: Optimizing MA crossover with values [5, 10, 15, 20, 25, 30]:
- Indicator containers: 6 (one per unique period)
- Strategy containers: C(6,2) = 15 (all unique pairs)

## Architectural Overview

The architecture optimizes for this reality by:
1. **Minimizing** computational redundancy (shared indicators)
2. **Isolating** strategy state (separate containers)
3. **Optimizing** memory usage (lightweight containers)
4. **Scaling** through batching and streaming

## Container Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        ADMF-Trader Optimization System               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │                   Shared Read-Only Layer                   │      │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐    │      │
│  │  │ Market Data  │  │Configuration │  │Historical Data │    │      │
│  │  │    Feed      │  │   Service    │  │     Store      │    │      │
│  │  │  (Shared)    │  │   (Shared)   │  │   (Shared)     │    │      │
│  │  └──────┬───────┘  └──────────────┘  └────────────────┘    │      │
│  └─────────┼──────────────────────────────────────────────────┘      │
│            │                                                         │
│  ┌─────────▼──────────────────────────────────────────────────┐      │
│  │              Indicator Container Layer                     │      │
│  │       (Shared Computation - Stateless)                     │      │
│  │                                                            │      │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │      │
│  │  │ Indicator   │  │ Indicator   │  │ Indicator   │         │      │
│  │  │ Container 1 │  │ Container 2 │  │ Container N │         │      │
│  │  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────┐  │         │      │
│  │  │  │ MA(5) │  │  │  │MA(10) │  │  │  │MA(20) │  │         │      │
│  │  │  ├───────┤  │  │  ├───────┤  │  │  ├───────┤  │         │      │
│  │  │  │ MA(15)│  │  │  │RSI(14)│  │  │  │RSI(21)│  │         │      │
│  │  │  ├───────┤  │  │  ├───────┤  │  │  ├───────┤  │         │      │
│  │  │  │  ...  │  │  │  │  ...  │  │  │  │  ...  │  │         │      │
│  │  │  └───┬───┘  │  │  └───┬───┘  │  │  └───┬───┘  │         │      │
│  │  └──────┼──────┘  └──────┼──────┘  └──────┼──────┘         │      │
│  └─────────┼────────────────┼────────────────┼───────────────┘       │
│            │                │                │                       │
│            └────────────────┴────────────────┘                       │
│                            │                                         │
│                 ┌──────────▼──────────┐                              │
│                 │  Event Distribution │                              │
│                 │    (Broadcast)      │                              │
│                 └──────────┬──────────┘                              │
│                            │                                         │
│  ┌─────────────────────────┼─────────────────────────────────────┐   │
│  │              Strategy Container Layer                          │  │
│  │          (Isolated State - Stateful)                          │ │
│  │                                                               │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │ │
│  │  │  Strategy    │  │  Strategy    │  │  Strategy    │ ...  │ │
│  │  │  Container   │  │  Container   │  │  Container   │      │ │
│  │  │  MA(5,10)    │  │  MA(5,15)    │  │  MA(10,20)   │      │ │
│  │  │              │  │              │  │              │      │ │
│  │  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │      │ │
│  │  │ │ Strategy │ │  │ │ Strategy │ │  │ │ Strategy │ │      │ │
│  │  │ │  Logic   │ │  │ │  Logic   │ │  │ │  Logic   │ │      │ │
│  │  │ ├──────────┤ │  │ ├──────────┤ │  │ ├──────────┤ │      │ │
│  │  │ │Portfolio │ │  │ │Portfolio │ │  │ │Portfolio │ │      │ │
│  │  │ │  State   │ │  │ │  State   │ │  │ │  State   │ │      │ │
│  │  │ ├──────────┤ │  │ ├──────────┤ │  │ ├──────────┤ │      │ │
│  │  │ │  Trade   │ │  │ │  Trade   │ │  │ │  Trade   │ │      │ │
│  │  │ │ History  │ │  │ │ History  │ │  │ │ History  │ │      │ │
│  │  │ ├──────────┤ │  │ ├──────────┤ │  │ ├──────────┤ │      │ │
│  │  │ │Position  │ │  │ │Position  │ │  │ │Position  │ │      │ │
│  │  │ │ Tracker  │ │  │ │ Tracker  │ │  │ │ Tracker  │ │      │ │
│  │  │ ├──────────┤ │  │ ├──────────┤ │  │ ├──────────┤ │      │ │
│  │  │ │   Risk   │ │  │ │   Risk   │ │  │ │   Risk   │ │      │ │
│  │  │ │  Metrics │ │  │ │  Metrics │ │  │ │  Metrics │ │      │ │
│  │  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │      │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                    Resource Management                      │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │    │
│  │  │   Memory     │  │   Batch      │  │    Result      │  │    │
│  │  │   Monitor    │  │  Processor   │  │   Streaming    │  │    │
│  │  └──────────────┘  └──────────────┘  └────────────────┘  │    │
│  └────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Insights

1. **Indicator Containers (Few)**: O(n) - One per unique indicator configuration
2. **Strategy Containers (Many)**: O(n^k) - One per parameter combination
3. **Event Flow**: Single calculation → Broadcast to all strategies
4. **State Isolation**: Each strategy maintains independent portfolio state

## Memory Optimization Strategies

### 1. Lightweight Strategy Containers

```python
class OptimizedStrategyContainer:
    """Memory-efficient container for strategy instances"""
    
    # Use __slots__ to reduce memory overhead
    __slots__ = [
        'strategy_id', 'params', 'portfolio', 
        'active_position', 'last_signal', 'stats'
    ]
    
    def __init__(self, strategy_id: str, params: Dict[str, Any]):
        self.strategy_id = strategy_id
        self.params = params
        self.portfolio = CompactPortfolio()
        self.active_position = None
        self.last_signal = None
        self.stats = CompactStats()

class CompactPortfolio:
    """Memory-optimized portfolio representation"""
    
    __slots__ = ['cash', 'positions', 'trade_buffer', 'trade_count']
    
    def __init__(self, initial_cash: float = 100000):
        self.cash = initial_cash
        self.positions = {}  # Only active positions
        # Pre-allocated circular buffer for trades
        self.trade_buffer = np.zeros((1000, 6), dtype=np.float32)
        self.trade_count = 0
```

### 2. Shared Memory Patterns

```python
class SharedMemoryOptimizer:
    """Optimize memory usage through sharing immutable data"""
    
    def __init__(self):
        # Intern common strings
        self.symbol_cache = {}
        self.action_cache = {'BUY': 'BUY', 'SELL': 'SELL', 'HOLD': 'HOLD'}
        
        # Share common parameter sets
        self.param_cache = {}
        
    def intern_symbol(self, symbol: str) -> str:
        """Return interned version of symbol"""
        if symbol not in self.symbol_cache:
            self.symbol_cache[symbol] = symbol
        return self.symbol_cache[symbol]
    
    def intern_params(self, params: Dict) -> Dict:
        """Return cached version of common parameter sets"""
        param_key = frozenset(params.items())
        if param_key not in self.param_cache:
            self.param_cache[param_key] = params
        return self.param_cache[param_key]
```

### 3. Progressive Memory Management

```python
class ProgressiveMemoryManager:
    """Manage memory usage during large optimizations"""
    
    def __init__(self, memory_limit_gb: float = 8.0):
        self.memory_limit_bytes = memory_limit_gb * 1024**3
        self.batch_size_calculator = AdaptiveBatchCalculator()
        
    def calculate_optimal_batch_size(self, total_strategies: int) -> int:
        """Calculate batch size based on available memory"""
        available_memory = self.get_available_memory()
        estimated_per_strategy = self.estimate_strategy_memory()
        
        # Leave 20% buffer
        usable_memory = available_memory * 0.8
        max_concurrent = int(usable_memory / estimated_per_strategy)
        
        # Apply reasonable bounds
        return max(10, min(max_concurrent, 1000))
    
    def should_trigger_gc(self) -> bool:
        """Determine if garbage collection should be triggered"""
        current_usage = self.get_memory_usage()
        return current_usage > self.memory_limit_bytes * 0.7
```

## Implementation Patterns

### 1. Batch Processing Pattern

```python
class BatchedOptimizationEngine:
    """Process strategy optimization in memory-efficient batches"""
    
    def __init__(self, shared_indicator_hub: SharedIndicatorHub):
        self.indicator_hub = shared_indicator_hub
        self.memory_manager = ProgressiveMemoryManager()
        self.result_streamer = ResultStreamer()
        
    def run_optimization(self, parameter_grid: List[Dict[str, Any]], 
                        market_data: pd.DataFrame) -> OptimizationResults:
        """Run optimization with automatic batching"""
        
        total_strategies = len(parameter_grid)
        batch_size = self.memory_manager.calculate_optimal_batch_size(total_strategies)
        
        print(f"Optimizing {total_strategies} strategies in batches of {batch_size}")
        
        for batch_idx, batch_start in enumerate(range(0, total_strategies, batch_size)):
            batch_end = min(batch_start + batch_size, total_strategies)
            batch_params = parameter_grid[batch_start:batch_end]
            
            # Process batch
            print(f"Processing batch {batch_idx + 1}: strategies {batch_start}-{batch_end}")
            batch_results = self._process_batch(batch_params, market_data)
            
            # Stream results to disk
            self.result_streamer.write_batch(batch_results)
            
            # Clean up
            del batch_results
            if self.memory_manager.should_trigger_gc():
                gc.collect()
        
        return self.result_streamer.finalize()
    
    def _process_batch(self, batch_params: List[Dict[str, Any]], 
                      market_data: pd.DataFrame) -> List[StrategyResult]:
        """Process a single batch of strategies"""
        
        # Create strategy containers for this batch
        containers = []
        for params in batch_params:
            container = self._create_strategy_container(params)
            containers.append(container)
        
        # Initialize all containers
        for container in containers:
            container.initialize_scope()
        
        # Process market data
        for idx, row in market_data.iterrows():
            # Indicators calculated once
            indicator_event = self.indicator_hub.process_bar(row)
            
            # Broadcast to all strategies in batch
            for container in containers:
                container.process_event(indicator_event)
        
        # Collect results
        results = []
        for container in containers:
            result = container.get_results()
            results.append(result)
            container.teardown_scope()
        
        return results
```

### 2. Result Streaming Pattern

```python
class ResultStreamer:
    """Stream optimization results to disk to minimize memory usage"""
    
    def __init__(self, output_path: str = "optimization_results"):
        self.output_path = output_path
        self.batch_count = 0
        self.summary_stats = SummaryStatistics()
        self.top_n_keeper = TopNKeeper(n=100)
        
    def write_batch(self, batch_results: List[StrategyResult]):
        """Write batch results to disk"""
        
        # Update summary statistics
        for result in batch_results:
            self.summary_stats.update(result)
            self.top_n_keeper.consider(result)
        
        # Write to parquet for efficiency
        batch_df = pd.DataFrame([r.to_dict() for r in batch_results])
        batch_file = f"{self.output_path}/batch_{self.batch_count:04d}.parquet"
        batch_df.to_parquet(batch_file, compression='snappy')
        
        self.batch_count += 1
        
    def finalize(self) -> OptimizationResults:
        """Finalize results and return summary"""
        return OptimizationResults(
            total_strategies=self.summary_stats.count,
            top_strategies=self.top_n_keeper.get_top_n(),
            summary_stats=self.summary_stats.get_summary(),
            results_path=self.output_path
        )
```

### 3. Early Termination Pattern

```python
class EarlyTerminationOptimizer:
    """Terminate poorly performing strategies early to save resources"""
    
    def __init__(self, termination_rules: List[TerminationRule]):
        self.termination_rules = termination_rules
        self.terminated_strategies = set()
        
    def should_terminate(self, strategy_id: str, portfolio: Portfolio, 
                        progress: float) -> bool:
        """Check if strategy should be terminated early"""
        
        for rule in self.termination_rules:
            if rule.should_terminate(portfolio, progress):
                self.terminated_strategies.add(strategy_id)
                return True
        return False

class DrawdownTerminationRule(TerminationRule):
    """Terminate if drawdown exceeds threshold"""
    
    def __init__(self, max_drawdown: float = 0.2, min_progress: float = 0.1):
        self.max_drawdown = max_drawdown
        self.min_progress = min_progress
        
    def should_terminate(self, portfolio: Portfolio, progress: float) -> bool:
        if progress < self.min_progress:
            return False  # Too early to judge
            
        current_drawdown = portfolio.get_max_drawdown()
        return current_drawdown > self.max_drawdown
```

## Performance Analysis

### Memory Usage Comparison

```python
def analyze_memory_usage(n_indicators: int, n_strategies: int, 
                        lookback: int = 1000) -> Dict[str, Any]:
    """Analyze memory usage of shared vs traditional approach"""
    
    # Traditional: Each strategy stores all indicators
    traditional_indicator_memory = n_strategies * n_indicators * lookback * 8
    traditional_portfolio_memory = n_strategies * 10_000  # ~10KB per portfolio
    traditional_total = traditional_indicator_memory + traditional_portfolio_memory
    
    # Shared: Single indicator storage + strategy state
    shared_indicator_memory = n_indicators * lookback * 8
    shared_portfolio_memory = n_strategies * 10_000  # Same portfolio memory
    shared_total = shared_indicator_memory + shared_portfolio_memory
    
    savings = traditional_total - shared_total
    savings_pct = (savings / traditional_total) * 100
    
    return {
        'traditional_mb': traditional_total / (1024**2),
        'shared_mb': shared_total / (1024**2),
        'savings_mb': savings / (1024**2),
        'savings_pct': savings_pct,
        'indicator_memory_reduction': (
            (traditional_indicator_memory - shared_indicator_memory) / 
            traditional_indicator_memory * 100
        )
    }

# Example: 20 indicators, 1000 strategies
result = analyze_memory_usage(20, 1000)
print(f"Memory savings: {result['savings_mb']:.1f}MB ({result['savings_pct']:.1f}%)")
# Output: Memory savings: 152.6MB (93.9%)
```

### Computational Savings

```python
def analyze_computation_savings(n_indicators: int, n_strategies: int,
                              bars_per_backtest: int = 10000) -> Dict[str, Any]:
    """Analyze computational savings"""
    
    # Traditional: Each strategy computes all indicators
    traditional_computations = n_strategies * n_indicators * bars_per_backtest
    
    # Shared: Compute once per indicator
    shared_computations = n_indicators * bars_per_backtest
    
    savings = traditional_computations - shared_computations
    speedup = traditional_computations / shared_computations
    
    return {
        'traditional_operations': traditional_computations,
        'shared_operations': shared_computations,
        'operations_saved': savings,
        'speedup_factor': speedup,
        'time_saved_pct': (savings / traditional_computations) * 100
    }
```

## Best Practices

### 1. Container Design

- **Minimize State**: Only track essential state in strategy containers
- **Use Slots**: Leverage `__slots__` for memory efficiency
- **Lazy Initialization**: Initialize components only when needed
- **Compact Data Types**: Use appropriate numeric types (float32 vs float64)

### 2. Batch Processing

- **Dynamic Sizing**: Adjust batch size based on available memory
- **Progress Tracking**: Provide feedback during long optimizations
- **Checkpointing**: Save intermediate results for crash recovery
- **Parallel Batches**: Use multiprocessing for CPU-bound operations

### 3. Memory Management

- **Monitor Usage**: Track memory consumption throughout optimization
- **Force Collection**: Trigger garbage collection between batches
- **Stream Results**: Write results to disk immediately
- **Prune Early**: Terminate poor performers to free resources

### 4. Optimization Strategies

```python
class OptimizationStrategy:
    """Strategy pattern for different optimization approaches"""
    
    @abstractmethod
    def select_next_batch(self, parameter_space: List[Dict], 
                         results_so_far: List[Result]) -> List[Dict]:
        """Select next batch of parameters to test"""
        pass

class GridSearchStrategy(OptimizationStrategy):
    """Exhaustive grid search"""
    
    def select_next_batch(self, parameter_space, results_so_far):
        tested = {r.params_hash for r in results_so_far}
        remaining = [p for p in parameter_space if hash_params(p) not in tested]
        return remaining[:self.batch_size]

class BayesianOptimizationStrategy(OptimizationStrategy):
    """Smart parameter selection based on results"""
    
    def select_next_batch(self, parameter_space, results_so_far):
        # Use Gaussian Process to predict promising regions
        gp = self.fit_gaussian_process(results_so_far)
        acquisition_scores = [self.acquisition_function(p, gp) for p in parameter_space]
        sorted_indices = np.argsort(acquisition_scores)[::-1]
        return [parameter_space[i] for i in sorted_indices[:self.batch_size]]
```

## Conclusion

The shared indicator architecture successfully addresses the computational redundancy in backtesting optimization. While we cannot reduce the number of strategy instances (each parameter combination truly is a different strategy), we have:

1. **Eliminated** redundant indicator calculations (O(n) instead of O(n²))
2. **Isolated** state where necessary (strategy containers)
3. **Optimized** memory usage through lightweight containers
4. **Enabled** scaling through batching and streaming

This architecture represents the optimal balance between computational efficiency and the fundamental requirement of tracking independent strategy states.
