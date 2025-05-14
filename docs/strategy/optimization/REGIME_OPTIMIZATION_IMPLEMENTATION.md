# Regime-Based Optimization Implementation Guide

## Overview

This document outlines the implementation approach for regime-based optimization, a technique that enhances trading strategy performance by optimizing parameters separately for different market regimes and enabling adaptive parameter selection based on the current detected regime.

Unlike traditional optimization that produces a single set of parameters, regime-based optimization produces multiple parameter sets optimized for specific market conditions, allowing strategies to adapt their behavior as market regimes change.

## Core Components

### 1. Regime Detection

Market regimes are distinct states of market behavior with different statistical properties. Common regime classifications include:

- **Volatility-based**: High/Medium/Low volatility periods
- **Trend-based**: Trending/Ranging/Choppy markets
- **Correlation-based**: High/Low cross-asset correlation environments
- **Liquidity-based**: High/Low liquidity periods
- **Macro-based**: Expansion/Contraction/Crisis economic environments

#### RegimeDetector Implementation

```python
class RegimeDetector(BaseComponent):
    """
    Detects market regimes based on configurable indicators and thresholds.
    """
    
    def __init__(self, name, config_loader, event_bus, component_config_key):
        super().__init__(name, config_loader, component_config_key)
        self._event_bus = event_bus
        
        # Configure regime detection parameters
        self._regime_indicators = {}
        self._regime_thresholds = self.get_specific_config("regime_thresholds", {})
        self._current_regime = "default"
        self._regime_history = []
        
        # Stabilization parameters to prevent rapid regime switching
        self._min_regime_duration = self.get_specific_config("min_regime_duration", 5)
        self._current_regime_duration = 0
        self._pending_regime = None
        
    def setup(self):
        """Initialize indicators and subscribe to events."""
        self._setup_regime_indicators()
        self._event_bus.subscribe(EventType.BAR, self.on_bar)
        self.state = BaseComponent.STATE_INITIALIZED
        
    def _setup_regime_indicators(self):
        """Initialize indicators used for regime detection."""
        indicator_configs = self.get_specific_config("indicators", {})
        
        for indicator_name, config in indicator_configs.items():
            indicator_type = config.get("type")
            params = config.get("parameters", {})
            
            if indicator_type == "volatility":
                self._regime_indicators[indicator_name] = VolatilityIndicator(**params)
            elif indicator_type == "trend_strength":
                self._regime_indicators[indicator_name] = TrendStrengthIndicator(**params)
            elif indicator_type == "correlation":
                self._regime_indicators[indicator_name] = CorrelationIndicator(**params)
            # Add more indicators as needed
    
    def get_current_regime(self):
        """Returns the current detected market regime."""
        return self._current_regime
        
    def detect_regime(self, market_data):
        """
        Detect the current market regime using configured indicators.
        Returns a regime identifier string and a boolean indicating if regime changed.
        """
        # Update all indicators with new market data
        for indicator in self._regime_indicators.values():
            indicator.update(market_data)
            
        # Get indicator values
        indicator_values = {name: indicator.value 
                           for name, indicator in self._regime_indicators.items()
                           if indicator.ready}
                           
        # Apply regime classification rules
        detected_regime = self._classify_regime(indicator_values)
        
        # Apply regime change stabilization
        if detected_regime != self._current_regime:
            if self._pending_regime is None:
                # Start tracking a potential regime change
                self._pending_regime = detected_regime
                self._pending_duration = 1
            elif detected_regime == self._pending_regime:
                # Continue tracking the same pending regime
                self._pending_duration += 1
                
                # If pending regime has been stable for sufficient duration, change the current regime
                if self._pending_duration >= self._min_regime_duration:
                    previous_regime = self._current_regime
                    self._current_regime = self._pending_regime
                    self._pending_regime = None
                    self._current_regime_duration = 0
                    
                    # Record regime change
                    self._regime_history.append({
                        'timestamp': market_data.get('timestamp'),
                        'previous_regime': previous_regime,
                        'new_regime': self._current_regime
                    })
                    
                    return self._current_regime, True  # Regime changed
            else:
                # Reset pending regime if detection is inconsistent
                self._pending_regime = detected_regime
                self._pending_duration = 1
        else:
            # No change in detected regime
            self._pending_regime = None
            self._current_regime_duration += 1
            
        return self._current_regime, False  # No regime change
    
    def _classify_regime(self, indicator_values):
        """Apply rules to classify the current regime based on indicator values."""
        # Simple threshold-based classification
        if not indicator_values or not self._regime_thresholds:
            return "default"
            
        # Check each defined regime
        for regime_name, thresholds in self._regime_thresholds.items():
            matches_all = True
            
            for indicator_name, threshold_config in thresholds.items():
                if indicator_name not in indicator_values:
                    matches_all = False
                    break
                    
                value = indicator_values[indicator_name]
                min_val = threshold_config.get("min")
                max_val = threshold_config.get("max")
                
                if (min_val is not None and value < min_val) or \
                   (max_val is not None and value > max_val):
                    matches_all = False
                    break
            
            if matches_all:
                return regime_name
                
        return "default"  # Default regime if no specific regime matched
        
    def on_bar(self, event):
        """Process bar event to update regime detection."""
        bar_data = event.get_data()
        
        # Detect regime and check if changed
        current_regime, regime_changed = self.detect_regime(bar_data)
        
        # Publish regime change event if needed
        if regime_changed:
            self.logger.info(f"Regime change detected: {self._regime_history[-1]['previous_regime']} -> {current_regime}")
            
            regime_change_event = Event(EventType.REGIME_CHANGE, {
                'timestamp': bar_data.get('timestamp'),
                'previous_regime': self._regime_history[-1]['previous_regime'],
                'new_regime': current_regime,
                'indicator_values': {name: indicator.value 
                                    for name, indicator in self._regime_indicators.items()
                                    if indicator.ready}
            })
            
            self._event_bus.publish(regime_change_event)
```

### 2. Enhanced Performance Tracking

To enable regime-based optimization, we need to track performance metrics by regime. This is implemented by enhancing the Portfolio or PerformanceTracker component:

```python
class RegimeAwarePerformanceTracker(BaseComponent):
    """
    Tracks performance metrics separately for each detected market regime.
    """
    
    def __init__(self, name, config_loader, event_bus, component_config_key, regime_detector_key="regime_detector"):
        super().__init__(name, config_loader, component_config_key)
        self._event_bus = event_bus
        self._regime_detector_key = regime_detector_key
        self._regime_detector = None
        
        # Performance metrics by regime
        self._metrics_by_regime = {}
        self._trades_by_regime = {}
        
    def setup(self):
        """Initialize and connect to regime detector."""
        # Get regime detector from container
        self._regime_detector = self._container.resolve(self._regime_detector_key)
        
        # Subscribe to trade events
        self._event_bus.subscribe(EventType.FILL, self.on_fill)
        self.state = BaseComponent.STATE_INITIALIZED
        
    def on_fill(self, event):
        """Record trade with associated regime."""
        fill_data = event.get_data()
        
        # Get current regime
        current_regime = self._regime_detector.get_current_regime()
        
        # Initialize regime data if not exists
        if current_regime not in self._trades_by_regime:
            self._trades_by_regime[current_regime] = []
            self._metrics_by_regime[current_regime] = {
                'total_return': 0.0,
                'win_count': 0,
                'loss_count': 0,
                'total_trades': 0
            }
        
        # Add trade to regime-specific list
        trade_record = fill_data.copy()
        trade_record['regime'] = current_regime
        self._trades_by_regime[current_regime].append(trade_record)
        
        # Update metrics for this regime
        # (Implementation depends on how trades/returns are tracked in your system)
        
    def get_metrics_by_regime(self):
        """Returns performance metrics grouped by regime."""
        return self._metrics_by_regime
        
    def get_trades_by_regime(self):
        """Returns all trades grouped by regime."""
        return self._trades_by_regime
```

### 3. Regime-Based Optimizer

The optimizer extends your existing optimization framework to analyze performance by regime:

```python
class RegimeBasedOptimizer(BasicOptimizer):
    """
    Extends basic optimizer to find optimal parameters for each market regime.
    """
    
    def __init__(self, name, config_loader, event_bus, component_config_key, container):
        super().__init__(name, config_loader, event_bus, component_config_key, container)
        
        # Configuration
        self._regimes = self.get_specific_config("regimes", ["default"])
        self._performance_tracker_key = self.get_specific_config("performance_tracker_key", "performance_tracker")
        
        # Results storage
        self._regime_params = {regime: None for regime in self._regimes}
        self._regime_metrics = {regime: None for regime in self._regimes}
        
    def run_regime_optimization(self):
        """
        Run standard optimization but analyze results by regime.
        """
        self.logger.info(f"Starting regime-based optimization for regimes: {self._regimes}")
        
        # Run standard grid search to test all parameter combinations
        all_results = self.run_grid_search()
        
        # Get performance tracker
        performance_tracker = self._container.resolve(self._performance_tracker_key)
        
        # For each parameter set tested, analyze performance by regime
        best_params_by_regime = {}
        best_metrics_by_regime = {}
        
        for params, metrics in all_results:
            # Get trades that occurred with these parameters
            trades_by_regime = performance_tracker.get_trades_by_regime()
            
            # For each regime, calculate performance metrics
            for regime in self._regimes:
                if regime not in trades_by_regime:
                    continue
                    
                regime_trades = trades_by_regime[regime]
                if not regime_trades:
                    continue
                    
                # Calculate performance for this regime
                regime_metrics = self._calculate_regime_metrics(regime_trades)
                
                # Check if this is the best so far for this regime
                if (regime not in best_metrics_by_regime or 
                    regime_metrics[self._optimization_metric] > best_metrics_by_regime[regime][self._optimization_metric]):
                    best_params_by_regime[regime] = params
                    best_metrics_by_regime[regime] = regime_metrics
        
        # Store results
        self._regime_params = best_params_by_regime
        self._regime_metrics = best_metrics_by_regime
        
        # Log results
        for regime in best_params_by_regime:
            self.logger.info(f"Best parameters for regime {regime}: {best_params_by_regime[regime]}")
            self.logger.info(f"Metrics: {best_metrics_by_regime[regime]}")
        
        # Save parameters to file
        self._save_regime_parameters()
        
        return best_params_by_regime
        
    def _calculate_regime_metrics(self, regime_trades):
        """Calculate performance metrics for a set of trades."""
        # Implementation depends on your specific metrics
        # Return a dictionary of metrics
        pass
        
    def _save_regime_parameters(self):
        """Save optimal parameters for each regime to a file."""
        output_file = self.get_specific_config("output_file", "regime_parameters.yaml")
        
        try:
            with open(output_file, 'w') as f:
                yaml.dump({
                    "regime_parameters": self._regime_params
                }, f)
            self.logger.info(f"Saved regime parameters to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving regime parameters: {e}")
```

### 4. Regime-Aware Strategy

The strategy component adapts its behavior based on the current detected regime:

```python
class RegimeAwareStrategy(BaseStrategy):
    """
    Strategy that adapts its parameters based on the current detected market regime.
    """
    
    def __init__(self, name, config_loader, event_bus, component_config_key, 
                 regime_detector_key="regime_detector"):
        super().__init__(name, config_loader, event_bus, component_config_key)
        
        # Regime configuration
        self._regime_detector_key = regime_detector_key
        self._regime_detector = None
        self._regime_parameters = {}
        self._current_regime = "default"
        self._default_parameters = None
        
        # Load regime-specific parameters
        self._load_regime_parameters()
        
    def setup(self):
        """Set up the strategy and connect to regime detector."""
        super().setup()
        
        # Get regime detector
        self._regime_detector = self._container.resolve(self._regime_detector_key)
        
        # Subscribe to regime change events
        self._event_bus.subscribe(EventType.REGIME_CHANGE, self._on_regime_change)
        
        # Store default parameters
        if hasattr(self, "get_parameters"):
            self._default_parameters = self.get_parameters()
            
    def _load_regime_parameters(self):
        """Load regime-specific parameters from configuration."""
        # Try loading from config first
        self._regime_parameters = self.get_specific_config("regime_parameters", {})
        
        # If no config parameters, try loading from file
        if not self._regime_parameters:
            regime_params_file = self.get_specific_config("regime_parameters_file", None)
            if regime_params_file:
                try:
                    with open(regime_params_file, 'r') as f:
                        params_data = yaml.safe_load(f)
                        
                    if "regime_parameters" in params_data:
                        self._regime_parameters = params_data["regime_parameters"]
                        self.logger.info(f"Loaded regime parameters for {len(self._regime_parameters)} regimes from {regime_params_file}")
                except Exception as e:
                    self.logger.error(f"Error loading regime parameters from {regime_params_file}: {e}")
        
        # Ensure we have default parameters
        if "default" not in self._regime_parameters and self._default_parameters:
            self._regime_parameters["default"] = self._default_parameters
            
    def _on_regime_change(self, event):
        """Handle regime change events."""
        regime_data = event.get_data()
        new_regime = regime_data.get("new_regime")
        
        if new_regime == self._current_regime:
            return  # No change
            
        self.logger.info(f"Regime change detected: {self._current_regime} -> {new_regime}")
        
        # Update current regime
        self._current_regime = new_regime
        
        # Switch parameters if we have them for this regime
        if new_regime in self._regime_parameters:
            target_params = self._regime_parameters[new_regime]
            self.logger.info(f"Switching parameters for regime {new_regime}: {target_params}")
            self.set_parameters(target_params)
        else:
            self.logger.warning(f"No parameters defined for regime {new_regime}, using default")
            if "default" in self._regime_parameters:
                self.set_parameters(self._regime_parameters["default"])
                
    def _on_bar_event(self, event):
        """Override to check current regime before processing."""
        # Check current regime
        current_regime = self._regime_detector.get_current_regime()
        if current_regime != self._current_regime:
            self._current_regime = current_regime
            
            # Switch parameters if we have them for this regime
            if current_regime in self._regime_parameters:
                self.set_parameters(self._regime_parameters[current_regime])
        
        # Process bar with current parameters
        super()._on_bar_event(event)
```

## Implementation Steps

### 1. Adding Technical Indicators for Regime Detection

First, implement indicators that will be used for regime detection:

```python
class ATRIndicator(BaseIndicator):
    """
    Average True Range indicator for measuring market volatility.
    """
    def __init__(self, period=14):
        self.period = period
        self.values = []
        self.current_value = None
        
    def update(self, bar_data):
        # Calculate true range
        high = bar_data.get('high', 0)
        low = bar_data.get('low', 0)
        prev_close = self.values[-1] if self.values else high
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        
        self.values.append(tr)
        
        # Keep only necessary history
        if len(self.values) > self.period:
            self.values.pop(0)
            
        # Calculate ATR
        if len(self.values) == self.period:
            self.current_value = sum(self.values) / self.period
            
        return self.current_value
        
    @property
    def value(self):
        return self.current_value
        
    @property
    def ready(self):
        return self.current_value is not None
```

### 2. Setting Up Regime-Based Optimization

Configure your system to use the regime-based optimization:

```yaml
# config.yaml
# Regime detector configuration
regime_detector:
  # Volatility-based regimes
  indicators:
    volatility:
      type: "atr"
      parameters:
        period: 14
        
  # Regime classification thresholds  
  regime_thresholds:
    high_volatility:
      volatility:
        min: 0.02
        max: null
    medium_volatility:
      volatility:
        min: 0.01
        max: 0.02
    low_volatility:
      volatility:
        min: null
        max: 0.01
        
  # Stabilization to prevent rapid regime switching
  min_regime_duration: 5

# Regime-specific optimizer configuration
regime_optimizer:
  # Regimes to optimize for
  regimes: ["high_volatility", "medium_volatility", "low_volatility"]
  
  # Component dependencies
  performance_tracker_key: "performance_tracker"
  strategy_key: "strategy"
  
  # Output file for regime parameters
  output_file: "config/regime_parameters.yaml"
```

### 3. Extending Main Application

Update your main.py to support regime-based optimization:

```python
# Add this to your main.py or command handling
if args.regime_optimize:
    # Register regime-specific components
    container.register_type(
        "regime_detector",
        RegimeDetector,
        instance_name="regime_detector",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="regime_detector"
    )
    
    container.register_type(
        "regime_performance_tracker",
        RegimeAwarePerformanceTracker,
        instance_name="regime_performance_tracker",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="performance_tracker",
        regime_detector_key="regime_detector"
    )
    
    container.register_type(
        "regime_optimizer",
        RegimeBasedOptimizer,
        instance_name="regime_optimizer",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="regime_optimizer",
        container=container
    )
    
    # Resolve and run
    optimizer = container.resolve("regime_optimizer")
    optimizer.setup()
    optimizer.run_regime_optimization()
```

## Handling Strategy Signal and Regime Conflicts

When implementing a regime-aware strategy, you may encounter conflicts between the strategy's signals and the current regime. There are several approaches to handling these conflicts:

### Approach 1: Strategy Overrides Regime (Initial Implementation)

Let strategy signals operate independently, using regimes only for parameter optimization.

**Trade-offs:**
- **Pros:** Preserves strategy logic, maintains flexibility for counter-regime trades, simpler implementation
- **Cons:** Potentially larger drawdowns during adverse regimes

### Approach 2: Regime Overrides Strategy

The regime classification could override the strategy's signals, filtering out or reversing signals that contradict the detected regime.

**Trade-offs:**
- **Pros:** Better downside protection, cleaner performance attribution
- **Cons:** Higher opportunity cost when strategy correctly identifies counter-regime movements

### Approach 3: Regime-Specific Position Sizing

Allow strategy to generate signals, but modify position size based on regime agreement.

**Trade-offs:**
- **Pros:** Balances signal preservation with risk management, smoother equity curve
- **Cons:** Complex position sizing logic, may underperform in strong directional moves

### Decision: Configurable Approach

For maximum flexibility, make the conflict resolution approach configurable:

```yaml
regime_strategy_interaction:
  # Options: 'strategy_priority', 'regime_priority', 'position_sizing'
  conflict_resolution: "strategy_priority"
  
  # For position sizing approach
  position_sizing:
    aligned: 1.0      # Full position when strategy and regime align
    contradicting: 0.5 # Half position when they contradict
```

## Conclusion

This approach to regime-based optimization offers significant advantages:

1. **Simplicity**: The design keeps components focused on their core responsibilities without complex interactions or event manipulations.

2. **Data Integrity**: By maintaining the natural flow of data and adding regime tracking, we avoid biases that could affect optimization results.

3. **Performance Attribution**: Tracking trades and metrics by regime provides valuable insights into strategy behavior under different market conditions.

4. **Adaptability**: The regime-aware strategy can dynamically adjust its parameters to the current market regime, improving performance across varied conditions.

This implementation provides a practical path to enhancing your trading system with regime-based optimization, making it more robust and adaptable to changing market conditions.