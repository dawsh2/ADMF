#   ADMF Implementation Guide

This guide provides a pragmatic approach to implementing the ADMF trading system, with specific steps to avoid common pitfalls and establish a working system quickly.

##   Implementation Philosophy

Based on our discussions and the project goals, we're taking an **incremental, working-first approach** with these key principles:

1.  **Start with minimal working code** that demonstrates complete functionality
2.  **Maintain event-driven architecture** from the beginning to prevent lookahead bias
3.  **Refactor incrementally** from simple to complex architecture
4.  **Test each step** to maintain a working system throughout development
5.  **Use passthrough modes** to simplify complex components when needed

##   Phased Implementation Plan

This implementation plan is designed to deliver value at each phase while building toward a complete architecture.

###   Phase 1: Minimal Working System

**Goal**: Create a single-file, event-driven backtest system that works end-to-end.

1.  **Create `main.py`**: Implement a simple but complete event-driven backtest system
    -   Include minimal event system, MA crossover strategy, simple portfolio and broker

2.  **Features to include**:
    -   Event-driven architecture (BAR → SIGNAL → ORDER → FILL) to prevent lookahead bias
    -   Moving average crossover strategy calculation
    -   Portfolio tracking with position updates
    -   End-to-end functionality with equity curve output

3.  **Key components in this file**:
    -   `MinimalEventSystem`: Simple publish/subscribe mechanism
    -   `SimpleBacktester`: Orchestrates the process and loads data
    -   `SimpleMAStrategy`: Calculates moving averages and generates signals
    -   `SimpleBroker`: Executes orders with basic fill logic
    -   `SimplePortfolio`: Tracks positions and equity

    This creates a foundation that works correctly and maintains event-driven integrity.

###   Phase 2: Core Module Extraction

**Goal**: Extract the core module while maintaining a working system.

**Refined Phase Breakdown**: To make the refactoring process more manageable, Phase 2 is broken down into smaller sub-phases.

1.  **Extract event system**:

    ```
    src/core/events/
    ├──   __init__.py
    ├──   event_bus.py           # Extract from MinimalEventSystem
    └──   event_types.py         # Define standard event types
    ```

2.  **Extract component base class**:

    ```
    src/core/component/
    ├──   __init__.py
    └──   component.py               # Base component with lifecycle methods
    ```

3.  **Extract configuration system**:

    ```
    src/core/config/
    ├──   __init__.py
    ├──   config.py                  # Configuration system
    └──   parameter_store.py         # Parameter storage
    ```

    * **CI Introduction**:  Introduce a basic Continuous Integration (CI) pipeline to automate unit test execution and code quality checks. This helps maintain code quality during refactoring.
    * **Documentation**:  Document the extraction process and any changes made to the core components. Update implementation guides accordingly.

4.  **Extract dependency injection container**:

    ```
    src/core/container/
    ├──   __init__.py
    └──   container.py               # DI container
    ```

5.  **Integrate extracted core components into `main.py`**:

    -   Replace the minimal implementations with the fully featured Core module components.
    -   Ensure the system still works as expected after the extraction.

###   Phase 3: Module Implementation and Regime-Specific Optimization

**Goal**: Implement the remaining modules (Data, Strategy, Risk, Execution) and incorporate market regime detection and optimization.

1.  **Implement Data Module**:
    -   Focus on basic CSV data loading and train/test splitting first.
    -   Add more data sources and transformations later.

2.  **Implement Strategy Module**:
    -   Start with simple strategies and indicators.
    -   Build the strategy component framework for reusability.
    -   Implement regime detection for different market conditions.
    -   Create RegimeAdaptiveStrategy for dynamic parameter switching.

3.  **Implement Risk Module**:
    -   Implement basic position sizing and risk management.
    -   Add regime-specific performance tracking.
    -   Track trades that span multiple regimes (boundary trades).

4.  **Implement Execution Module**:
    -   Begin with a simulated broker for backtesting.
    -   Implement order management and slippage/commission models.

5.  **Implement Enhanced Optimization**:
    -   Develop regime-specific parameter optimization.
    -   Build framework for tracking performance by market regime.
    -   Optimize strategy parameters for each detected regime.
    -   Handle boundary trades that span multiple regimes.

    * **Granular Development**: Break down each module's implementation into smaller, iterative tasks.
    * **Testing**:  Prioritize writing unit and integration tests for each component as it's developed.
    * **Performance**:  Pay attention to potential performance bottlenecks, especially in the Data and Execution modules. Choose efficient data structures and algorithms.

###   Phase 4: Advanced Features and Optimization

**Goal**: Implement advanced features and optimize system performance.

1.  **Advanced Strategies**:
    -   Implement composite and ensemble strategies.
    -   Add more sophisticated indicators and rules.

2.  **Optimization Framework**:
    -   Implement parameter optimization techniques.
    -   Define robust objective functions.

3.  **Live Trading**:
    -   Integrate with live broker APIs.
    -   Implement real-time data feeds.
    -   Add robust error handling and monitoring.

4.  **Performance Tuning**:
    -   Profile and optimize system performance.
    -   Scale the system to handle increased data and trading volume.

    * **Documentation**:  Maintain comprehensive documentation throughout the development process. Document each module, class, and function.
    * **Code Reviews**:  Conduct regular code reviews to ensure code quality and consistency.

##   Additional Considerations

* **Communication**:  Maintain clear and consistent communication within the development team.
* **Version Control**:  Use a version control system (e.g., Git) effectively for code management and collaboration.
* **Flexibility**: Design the system to be flexible and adaptable to future changes and requirements.

##   Example: Regime-Adaptive Strategy (Phase 3)

The following illustrates how the system implements regime-specific optimization and parameter switching:

```python
class RegimeAdaptiveStrategy(Strategy):
    """
    Strategy that adapts its parameters based on detected market regime.
    """
    def __init__(self, name, base_strategy_class, regime_detector, parameters=None):
        super().__init__(name, parameters)
        self.base_strategy_class = base_strategy_class
        self.regime_detector = regime_detector
        self.regime_strategies = {}
        self.current_regime = "default"
        
    def setup(self):
        # Load optimized parameters for each regime
        regime_params = self.load_regime_parameters()
        
        # Create a strategy instance for each regime with optimized parameters
        for regime, params in regime_params.items():
            strategy = self.base_strategy_class(f"{self.name}_{regime}", params)
            self.regime_strategies[regime] = strategy
            
        # Subscribe to CLASSIFICATION events for regime changes
        self.event_bus.subscribe(EventType.CLASSIFICATION, self.on_classification)
    
    def on_classification(self, event):
        # Update current regime when classification changes
        new_regime = event.payload.get("classification")
        if new_regime and new_regime != self.current_regime:
            self.logger.info(f"Switching from regime '{self.current_regime}' to '{new_regime}'")
            self.current_regime = new_regime
    
    def on_bar(self, event):
        # Forward the bar event to the current regime's strategy
        regime = self.current_regime
        if regime not in self.regime_strategies:
            regime = "default"  # Fallback to default if no strategy for current regime
            
        # Get signal from the appropriate regime-specific strategy
        return self.regime_strategies[regime].on_bar(event)
```

##   Example: Enhanced Optimizer with Regime Support (Phase 3)

```python
class EnhancedOptimizer(BaseOptimizer):
    """
    Optimizer that conducts regime-specific parameter optimization.
    """
    def __init__(self, name, parameters=None):
        super().__init__(name, parameters)
        self.regime_performance = {}
        self.min_trades_per_regime = 5
        
    def optimize(self, strategy_class, parameter_grid, backtest_func):
        best_params_by_regime = {}
        regimes_encountered = set()
        
        for params in self.generate_parameter_combinations(parameter_grid):
            # Reset portfolio before each run
            self.reset_portfolio()
            
            # Run backtest with current parameters
            result = backtest_func(strategy_class, params)
            
            # Analyze performance by regime
            regime_metrics = result["portfolio"].get_regime_performance()
            
            # Update list of encountered regimes
            regimes_encountered.update(regime_metrics.keys())
            
            # Update best parameters for each regime
            for regime, metrics in regime_metrics.items():
                if metrics["trade_count"] >= self.min_trades_per_regime:
                    if regime not in best_params_by_regime or \
                       metrics["sharpe_ratio"] > best_params_by_regime[regime]["metric"]["value"]:
                        best_params_by_regime[regime] = {
                            "parameters": params.copy(),
                            "metric": {
                                "name": "sharpe_ratio",
                                "value": metrics["sharpe_ratio"],
                                "higher_is_better": True
                            }
                        }
        
        return {
            "regime_best_parameters": best_params_by_regime,
            "regimes_encountered": list(regimes_encountered)
        }
```

##   Example: Boundary Trade Handling (Phase 3)

```python
class BasicPortfolio(BaseComponent):
    # ... other portfolio methods ...
    
    def close_position(self, symbol, quantity, price, timestamp, regime=None):
        """Close a position with proper regime attribution."""
        if symbol not in self.open_positions:
            return None
            
        position = self.open_positions[symbol]
        entry_regime = position.get("regime", "default")
        exit_regime = regime or self._current_market_regime or "default"
        
        # Calculate P&L
        pnl = self._calculate_pnl(position, quantity, price)
        
        # Track as boundary trade if regimes differ
        is_boundary_trade = entry_regime != exit_regime
        trade_record = {
            "symbol": symbol,
            "entry_time": position["entry_time"],
            "entry_price": position["entry_price"],
            "exit_time": timestamp,
            "exit_price": price,
            "quantity": quantity,
            "pnl": pnl,
            "entry_regime": entry_regime,
            "exit_regime": exit_regime,
            "is_boundary_trade": is_boundary_trade
        }
        
        # Add to appropriate regime performance tracking
        if is_boundary_trade:
            transition_key = f"{entry_regime}_to_{exit_regime}"
            if transition_key not in self._boundary_trade_performance:
                self._boundary_trade_performance[transition_key] = {
                    "trades": [],
                    "total_pnl": 0.0,
                    "win_count": 0,
                    "loss_count": 0
                }
                
            self._boundary_trade_performance[transition_key]["trades"].append(trade_record)
            self._boundary_trade_performance[transition_key]["total_pnl"] += pnl
            if pnl > 0:
                self._boundary_trade_performance[transition_key]["win_count"] += 1
            else:
                self._boundary_trade_performance[transition_key]["loss_count"] += 1
        else:
            # Regular single-regime trade tracking
            if exit_regime not in self._regime_performance:
                self._regime_performance[exit_regime] = {
                    "trades": [],
                    "total_pnl": 0.0,
                    "win_count": 0,
                    "loss_count": 0
                }
                
            self._regime_performance[exit_regime]["trades"].append(trade_record)
            self._regime_performance[exit_regime]["total_pnl"] += pnl
            if pnl > 0:
                self._regime_performance[exit_regime]["win_count"] += 1
            else:
                self._regime_performance[exit_regime]["loss_count"] += 1
                
        # Update portfolio state
        self._update_portfolio_state_after_trade(symbol, quantity, price, pnl)
        return trade_record
```

##   Example: Ensemble Strategy (Phase 4)

To illustrate how the system can be extended, consider the implementation of an ensemble strategy:

```python
class EnsembleStrategy(Strategy):
    """
    Combines multiple strategies with weights.
    """

    def __init__(self, name, parameters=None):
        super().__init__(name, parameters)
        self.strategies = []
        self.weights = {}

    def add_strategy(self, strategy, weight):
        self.strategies.append(strategy)
        self.weights[strategy.name] = weight

    def on_bar(self, event):
        bar = event.get_data()
        signals = []
        for strategy in self.strategies:
            signal = strategy.on_bar(event)  # Get signal from each strategy
            if signal:
                signals.append({
                    'signal': signal,
                    'weight': self.weights[strategy.name]
                })

        final_signal = self._aggregate_signals(signals, bar)
        if final_signal:
            self.emit_signal(final_signal)

    def _aggregate_signals(self, signals, bar):
        """Aggregate signals based on weights."""
        #   Calculate weighted sum of signal strengths
        buy_strength = sum(s['weight'] for s in signals if s['signal']['direction'] == 'BUY' for s in signals if s['signal']['direction'] == 'BUY')
        sell_strength = sum(s['weight'] for s in signals if s['signal']['direction'] == 'SELL')

        #   Generate signal if strength exceeds threshold
        direction = None
        if buy_strength > 0.6:
            direction = 'BUY'
        elif sell_strength > 0.6:
            direction = 'SELL'

        if direction:
            return {
                'timestamp': bar['timestamp'],
                'symbol': bar['symbol'],
                'direction': direction,
                'price': bar['close'],
                'strength': buy_strength if direction == 'BUY' else sell_strength
            }
        return None

## Conclusion

This implementation guide provides a pragmatic approach to building the ADMF trading system, starting with a minimal working system and refactoring toward a complete architecture. By following this approach, you can:

1. Have a working system at every stage of development
2. Maintain proper event-driven architecture to prevent lookahead bias
3. Add architectural complexity only as needed
4. Implement advanced features like ensemble strategies in a structured way

The key is to start simple, get something working, and incrementally improve while always maintaining a functioning system.
