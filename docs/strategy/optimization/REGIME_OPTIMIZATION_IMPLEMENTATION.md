# Regime-Based Optimization Implementation Guide

## Overview

This document outlines the implementation approach for regime-based optimization, a technique that enhances trading strategy performance by optimizing parameters separately for different market regimes and enabling adaptive parameter selection based on the current detected regime.

Unlike traditional optimization that produces a single set of parameters, regime-based optimization produces multiple parameter sets optimized for specific market conditions, allowing strategies to adapt their behavior as market regimes change.

## Architecture

The implementation follows a clean separation of concerns using the Classifier framework:

1. **Classification**: RegimeDetector (Classifier) identifies market regimes
2. **Performance Tracking**: Enhanced tracking records trades with regime information
3. **Optimization**: Optimizer analyzes performance by regime
4. **Adaptation**: RegimeAwareStrategy adapts parameters based on detected regime

This approach maintains the natural flow of data while adding regime awareness at key points in the system.

## Core Components

### 1. Regime Detection with Classifier

RegimeDetector extends the Classifier base class to identify market regimes:

```python
from ..core.classifier import Classifier
from typing import Dict, Any

class RegimeDetector(Classifier):
    """
    Detects market regimes based on configurable indicators and thresholds.
    """
    
    def __init__(self, instance_name, config_loader, event_bus, component_config_key=None):
        super().__init__(instance_name, config_loader, event_bus, component_config_key)
        
        # Configure regime detection parameters
        self._regime_indicators = {}
        self._regime_thresholds = self.get_specific_config("regime_thresholds", {})
        
        # Stabilization parameters to prevent rapid regime switching
        self._min_regime_duration = self.get_specific_config("min_regime_duration", 5)
        self._current_regime_duration = 0
        self._pending_regime = None
        self._pending_duration = 0
    
    def setup(self):
        """Initialize indicators and subscribe to events."""
        self._setup_regime_indicators()
        self._event_bus.subscribe("BAR", self.on_bar)
        self.state = self.STATE_INITIALIZED
    
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
            # Add more indicators as needed
    
    def classify(self, data: Dict[str, Any]) -> str:
        """
        Classify market data into a regime.
        
        Args:
            data: Market data to classify
            
        Returns:
            str: Regime label
        """
        # Update all indicators with new market data
        for indicator in self._regime_indicators.values():
            indicator.update(data)
        
        # Get indicator values
        indicator_values = {
            name: indicator.value 
            for name, indicator in self._regime_indicators.items()
            if getattr(indicator, "ready", True)
        }
        
        # Apply regime classification rules
        detected_regime = self._classify_regime(indicator_values)
        
        # Apply regime change stabilization
        final_regime, regime_changed = self._apply_stabilization(detected_regime)
        
        # If regime changed, publish an event
        if regime_changed:
            self._publish_regime_change_event(data, final_regime)
        
        return final_regime
    
    def _classify_regime(self, indicator_values: Dict[str, float]) -> str:
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
    
    def _apply_stabilization(self, detected_regime: str) -> (str, bool):
        """
        Apply stabilization to prevent rapid regime switching.
        
        Args:
            detected_regime: The newly detected regime
            
        Returns:
            tuple: (final_regime, regime_changed)
        """
        current_classification = self.get_current_classification() or "default"
        regime_changed = False
        
        # If no change, reset pending and increment duration
        if detected_regime == current_classification:
            self._pending_regime = None
            self._pending_duration = 0
            self._current_regime_duration += 1
            return current_classification, False
        
        # If new regime detected
        if self._pending_regime is None or detected_regime != self._pending_regime:
            # Start tracking a new pending regime
            self._pending_regime = detected_regime
            self._pending_duration = 1
            return current_classification, False
        
        # Continue tracking same pending regime
        self._pending_duration += 1
        
        # If pending regime has been stable for sufficient duration, change the regime
        if self._pending_duration >= self._min_regime_duration:
            previous_regime = current_classification
            self._current_regime_duration = 0
            self._pending_regime = None
            self._pending_duration = 0
            
            # Record regime change
            self._classification_history.append({
                'previous_classification': previous_regime,
                'new_classification': detected_regime,
                'changed': True
            })
            
            # Update current classification
            self._current_classification = detected_regime
            
            return detected_regime, True
        
        # Not stable enough yet, maintain current regime
        return current_classification, False
        
    def _publish_regime_change_event(self, data, new_regime):
        """Publish a regime change event."""
        from ..core.event import Event, EventType
        
        regime_change_event = Event(
            EventType.REGIME_CHANGE, 
            {
                'timestamp': data.get('timestamp'),
                'previous_regime': self._classification_history[-1]['previous_classification'],
                'new_regime': new_regime,
                'indicator_values': {
                    name: indicator.value 
                    for name, indicator in self._regime_indicators.items()
                    if getattr(indicator, "ready", True)
                }
            }
        )
        
        self._event_bus.publish(regime_change_event)
    
    def get_regime_data(self):
        """Get additional data about the current regime."""
        return {
            'regime': self.get_current_classification(),
            'duration': self._current_regime_duration,
            'indicators': {
                name: indicator.value 
                for name, indicator in self._regime_indicators.items()
                if getattr(indicator, "ready", True)
            }
        }
```

### 2. Enhanced Performance Tracking

To enable regime-based optimization, we track performance metrics by regime:

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
        self._active_trades = {}  # Symbol -> Trade info including entry regime
        
    def setup(self):
        """Initialize and connect to regime detector."""
        # Get regime detector from container
        self._regime_detector = self._container.resolve(self._regime_detector_key)
        
        # Subscribe to trade events
        self._event_bus.subscribe("FILL", self.on_fill)
        self.state = self.STATE_INITIALIZED
        
    def on_fill(self, event):
        """
        Record trade with associated regime.
        Properly attributes P&L to the specific regimes in which it occurred.
        """
        fill_data = event.get_data()
        symbol = fill_data.get('symbol')
        timestamp = fill_data.get('timestamp')
        quantity = fill_data.get('quantity', 0)
        price = fill_data.get('price', 0)
        direction = fill_data.get('direction', '')
        
        # Get current regime
        current_regime = self._regime_detector.get_current_classification()
        
        # Initialize regime data if not exists
        if current_regime not in self._trades_by_regime:
            self._trades_by_regime[current_regime] = []
            self._metrics_by_regime[current_regime] = {
                'total_return': 0.0,
                'win_count': 0,
                'loss_count': 0,
                'total_trades': 0
            }
        
        # Handle boundary case: trade spans multiple regimes
        # For position entry
        if self._is_entry(direction, quantity):
            # Record new position with entry regime
            self._active_trades[symbol] = {
                'entry_regime': current_regime,
                'entry_price': price,
                'entry_time': timestamp,
                'quantity': quantity,
                'direction': direction,
                'pnl_by_regime': {current_regime: 0.0}  # Track P&L by regime
            }
            
        # For position exit or partial exit
        elif self._is_exit(direction, quantity, symbol):
            if symbol in self._active_trades:
                trade_data = self._active_trades[symbol]
                entry_regime = trade_data['entry_regime']
                
                # Calculate total P&L for the trade
                entry_price = trade_data['entry_price']
                exit_price = price
                trade_quantity = min(abs(quantity), abs(trade_data['quantity']))
                
                if direction == "BUY":
                    pnl = (exit_price - entry_price) * trade_quantity * -1  # Short trade
                else:
                    pnl = (exit_price - entry_price) * trade_quantity  # Long trade
                
                # Attribute P&L appropriately
                if entry_regime == current_regime:
                    # Trade opened and closed in same regime - simple case
                    self._metrics_by_regime[current_regime]['total_return'] += pnl
                    
                    # Record complete trade
                    trade_record = {
                        'symbol': symbol,
                        'entry_time': trade_data['entry_time'],
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': trade_quantity,
                        'pnl': pnl,
                        'regime': current_regime
                    }
                    
                    self._trades_by_regime[current_regime].append(trade_record)
                    
                else:
                    # Trade spans multiple regimes - handle boundary case
                    # For this implementation, we attribute all P&L to the exit regime
                    # Note: A more sophisticated implementation could apportion P&L based on
                    # price movements during each regime
                    self._metrics_by_regime[current_regime]['total_return'] += pnl
                    
                    # Record as a boundary trade
                    trade_record = {
                        'symbol': symbol,
                        'entry_time': trade_data['entry_time'],
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': trade_quantity,
                        'pnl': pnl,
                        'entry_regime': entry_regime,
                        'exit_regime': current_regime,
                        'is_boundary_trade': True
                    }
                    
                    # Add to both regimes for analysis
                    if entry_regime in self._trades_by_regime:
                        self._trades_by_regime[entry_regime].append(trade_record)
                    self._trades_by_regime[current_regime].append(trade_record)
                
                # Update metrics
                if pnl > 0:
                    self._metrics_by_regime[current_regime]['win_count'] += 1
                else:
                    self._metrics_by_regime[current_regime]['loss_count'] += 1
                    
                self._metrics_by_regime[current_regime]['total_trades'] += 1
                
                # Remove or update the active trade
                if abs(quantity) >= abs(trade_data['quantity']):
                    del self._active_trades[symbol]
                else:
                    trade_data['quantity'] -= quantity
        
    def _is_entry(self, direction, quantity):
        """Determine if this is a position entry."""
        return (direction == "BUY" and quantity > 0) or (direction == "SELL" and quantity < 0)
        
    def _is_exit(self, direction, quantity, symbol):
        """Determine if this is a position exit."""
        if symbol not in self._active_trades:
            return False
            
        active_trade = self._active_trades[symbol]
        return ((direction == "SELL" and active_trade['direction'] == "BUY") or
                (direction == "BUY" and active_trade['direction'] == "SELL"))
        
    def get_metrics_by_regime(self):
        """Returns performance metrics grouped by regime."""
        return self._metrics_by_regime
        
    def get_trades_by_regime(self):
        """Returns all trades grouped by regime."""
        return self._trades_by_regime
        
    def get_boundary_trades(self):
        """Returns trades that spanned multiple regimes."""
        boundary_trades = []
        for regime_trades in self._trades_by_regime.values():
            for trade in regime_trades:
                if trade.get('is_boundary_trade', False):
                    boundary_trades.append(trade)
        return boundary_trades
```

### 3. Enhanced Optimizer

We extend the existing optimizer to perform regime-specific optimization:

```python
class EnhancedOptimizer(BaseOptimizer):
    """
    Extends basic optimizer to support regime-based parameter optimization.
    """
    
    def __init__(self, name, config_loader, event_bus, component_config_key, container):
        super().__init__(name, config_loader, event_bus, component_config_key, container)
        
        # Configuration
        self._use_regime_optimization = self.get_specific_config("use_regime_optimization", False)
        self._regimes = self.get_specific_config("regimes", ["default"])
        self._performance_tracker_key = self.get_specific_config("performance_tracker_key", "performance_tracker")
        self._min_regime_samples = self.get_specific_config("min_regime_samples", 30)
        
        # Results storage
        self._regime_params = {regime: None for regime in self._regimes}
        self._regime_metrics = {regime: None for regime in self._regimes}
        
    def run_optimization(self):
        """Enhanced optimization that supports regime-based optimization."""
        if self._use_regime_optimization:
            return self.run_regime_optimization()
        else:
            return super().run_optimization()
            
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
                    
                # Check for sufficient sample size
                if len(regime_trades) < self._min_regime_samples:
                    self.logger.warning(f"Insufficient samples for regime {regime}: {len(regime_trades)} trades. Min required: {self._min_regime_samples}")
                    continue
                    
                # Calculate performance for this regime
                regime_metrics = self._calculate_regime_metrics(regime_trades)
                
                # Check if this is the best so far for this regime
                if (regime not in best_metrics_by_regime or 
                    regime_metrics[self._optimization_metric] > best_metrics_by_regime[regime][self._optimization_metric]):
                    best_params_by_regime[regime] = params
                    best_metrics_by_regime[regime] = regime_metrics
        
        # Analyze boundary trades
        boundary_trades = performance_tracker.get_boundary_trades()
        if boundary_trades:
            self.logger.info(f"Found {len(boundary_trades)} trades that span multiple regimes")
            # Future enhancement: Analyze boundary trades to determine optimal handling
        
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
        pass
        
    def _save_regime_parameters(self):
        """Save optimal parameters for each regime to a file."""
        import yaml
        
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
        self._default_parameters = None
        
        # Load regime-specific parameters
        self._load_regime_parameters()
        
    def setup(self):
        """Set up the strategy and connect to regime detector."""
        super().setup()
        
        # Get regime detector
        self._regime_detector = self._container.resolve(self._regime_detector_key)
        
        # Subscribe to regime change events
        self._event_bus.subscribe("REGIME_CHANGE", self._on_regime_change)
        
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
                    import yaml
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
        
        self.logger.info(f"Regime change detected: {regime_data.get('previous_regime')} -> {new_regime}")
        
        # Switch parameters if we have them for this regime
        if new_regime in self._regime_parameters:
            target_params = self._regime_parameters[new_regime]
            self.logger.info(f"Switching parameters for regime {new_regime}: {target_params}")
            self.set_parameters(target_params)
        else:
            self.logger.warning(f"No parameters defined for regime {new_regime}, using default")
            if "default" in self._regime_parameters:
                self.set_parameters(self._regime_parameters["default"])
                
    def on_bar(self, event):
        """
        Override to check current regime before processing.
        Note: This implementation maintains positions through regime changes,
        only switching parameters. Signal generation continues to follow
        standard strategy logic, not directly influenced by regime.
        """
        # Get market data
        data = event.get_data()
        
        # Check current regime
        current_regime = self._regime_detector.get_current_classification()
        
        # Switch parameters if needed
        if current_regime in self._regime_parameters:
            self.set_parameters(self._regime_parameters[current_regime])
        
        # Process bar with current parameters using standard strategy logic
        # This implements Option 2: Strategy Overrides Regime - regimes are used
        # only for parameter optimization, not for signal filtering
        super().on_bar(event)
```

## Handling Trades at Regime Boundaries

For trades that span multiple regimes, the implementation follows these principles:

1. **Leave Positions Open**: We do not force closure of positions when regimes change
2. **Proper P&L Attribution**: We attribute P&L to the specific regimes in which it occurred
3. **Boundary Trade Tracking**: We mark trades that span multiple regimes and track them separately
4. **Statistical Analysis**: We collect data on boundary trades for future analysis

This approach allows us to:
- Maintain realistic trading behavior that matches live conditions
- Properly attribute performance to specific regimes
- Collect data for future refinement of the boundary handling strategy

For the initial implementation, we attribute all P&L to the exit regime, but this can be refined in future versions to apportion P&L based on price movements during each regime.

## Strategy Signal vs Regime Conflicts

When implementing a regime-aware strategy, conflicts between strategy signals and current regime are handled using one of these approaches:

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

### Decision: Strategy Overrides Regime with Boundary Trade Tracking

For our implementation, we've chosen **Approach 1: Strategy Overrides Regime** with enhanced boundary trade tracking. This strikes a balance between simplicity and effectiveness, while collecting data that may inform more sophisticated approaches in the future.

## Implementation Steps

### 1. Create Classifier Framework

First, establish the Classifier base class in the strategy module.

### 2. Implement RegimeDetector

Create the RegimeDetector class that extends Classifier and identifies market regimes.

### 3. Enhance Performance Tracking

Update performance tracking to record regime information.

### 4. Extend Optimizer

Modify the optimizer to analyze performance by regime.

### 5. Create RegimeAwareStrategy

Implement a strategy that adapts parameters based on regimes.

### 6. Configure System

Set up thresholds, indicators, and parameters for regime detection and optimization.

## Configuration Example

```yaml
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
    medium_volatility:
      volatility:
        min: 0.01
        max: 0.02
    low_volatility:
      volatility:
        max: 0.01
        
  # Stabilization to prevent rapid regime switching
  min_regime_duration: 5

# Optimizer configuration
optimizer:
  # Enable regime-based optimization
  use_regime_optimization: true
  
  # Regimes to optimize for
  regimes: ["high_volatility", "medium_volatility", "low_volatility"]
  
  # Minimum trades required for reliable optimization
  min_regime_samples: 30
  
  # Output file for regime parameters
  output_file: "config/regime_parameters.yaml"
```

## Conclusion

This implementation of regime-based optimization leverages the Classifier framework to create a clean, modular approach to market regime detection and parameter optimization. By maintaining clear separation of concerns while adding regime awareness at key points, we enable strategies to adapt to changing market conditions without compromising system integrity.

The approach properly handles boundary trades, maintains positions through regime transitions, and collects valuable data for future refinement. By starting with the "Strategy Overrides Regime" approach, we enable immediate benefits while laying the groundwork for more sophisticated regime-based trading in the future.