# Classifier Design for ADMF-Trader

## Overview

The Classifier framework provides a clean abstraction for components that analyze market data and produce categorical labels without directly generating trading signals. This document outlines the design, motivation, and implementation approach for classifiers in the ADMF-Trader system.

## Motivation

While developing the regime-based optimization system, we recognized the need for a distinct component type that:

1. Analyzes market data to identify conditions or states
2. Produces categorical labels (e.g., market regimes, volatility states)
3. Operates independently from signal generation
4. Provides contextual information for other components

Rather than forcing the RegimeDetector into the Strategy hierarchy or creating a one-off component, we're establishing a proper Classifier abstraction that can be reused for various classification needs.

## Design Philosophy

Classifiers follow these key principles:

1. **Separation of Concerns**: Classifiers identify market conditions but don't make trading decisions
2. **Composability**: Classifiers can be used by strategies or optimization components
3. **Standard Interface**: All classifiers follow a consistent interface
4. **Integration with Component Architecture**: Classifiers extend the BaseComponent system

## Implementation

The Classifier base class will be implemented as follows:

```python
# src/strategy/classifier.py
from abc import abstractmethod
from typing import Any, Dict, Optional

from ..core.component import BaseComponent

class Classifier(BaseComponent):
    """
    Base class for all market data classifiers.
    
    Classifiers analyze market data and produce categorical labels without
    directly generating trading signals. Examples include regime detectors,
    market state classifiers, and volatility classifiers.
    """
    
    def __init__(self, 
                instance_name: str, 
                config_loader, 
                event_bus,
                component_config_key: str = None):
        """
        Initialize the Classifier.
        
        Args:
            instance_name (str): The unique name of this classifier instance
            config_loader: Configuration loader instance
            event_bus: Event bus for publishing classification events
            component_config_key (str, optional): Configuration key for this component
        """
        super().__init__(instance_name, config_loader, component_config_key)
        self._event_bus = event_bus
        self._current_classification = None
        self._classification_history = []
    
    @abstractmethod
    def classify(self, data: Dict[str, Any]) -> str:
        """
        Classify market data into a categorical label.
        
        Args:
            data: Market data to classify
            
        Returns:
            str: Classification label
        """
        pass
    
    def get_current_classification(self) -> Optional[str]:
        """
        Returns the current classification label.
        
        Returns:
            str: Current classification label or None if not yet classified
        """
        return self._current_classification
    
    def get_classification_history(self) -> list:
        """
        Returns the history of classifications with timestamps.
        
        Returns:
            list: List of classification events with timestamps
        """
        return self._classification_history
    
    def setup(self):
        """Set up classifier resources."""
        self.logger.info(f"Setting up classifier '{self.name}'")
        # Subscribe to BAR events or other relevant data events
        if self._event_bus:
            self._event_bus.subscribe("BAR", self.on_bar)
        self.state = BaseComponent.STATE_INITIALIZED
    
    def on_bar(self, event):
        """
        Process bar event and update classification.
        
        Args:
            event: Bar event with market data
        """
        # Extract market data from event
        data = event.get_data()
        
        # Classify the data
        classification = self.classify(data)
        
        # Check if classification changed
        classification_changed = classification != self._current_classification
        
        # Update current classification
        self._current_classification = classification
        
        # Record in history
        self._classification_history.append({
            'timestamp': data.get('timestamp'),
            'classification': classification,
            'changed': classification_changed
        })
        
        # Emit classification event if changed
        if classification_changed and self._event_bus:
            self._event_bus.publish(self._create_classification_event(data, classification))
    
    def _create_classification_event(self, data, classification):
        """
        Create a classification event.
        
        Args:
            data: Market data that triggered the classification
            classification: The classification label
            
        Returns:
            Event object
        """
        from ..core.event import Event, EventType
        
        return Event(
            EventType.CLASSIFICATION,
            {
                'timestamp': data.get('timestamp'),
                'classifier': self.name,
                'classification': classification,
                'previous_classification': self._classification_history[-2]['classification'] 
                    if len(self._classification_history) > 1 else None
            }
        )
    
    def start(self):
        """Start the classifier."""
        self.logger.info(f"Starting classifier '{self.name}'")
        self.state = BaseComponent.STATE_STARTED
    
    def stop(self):
        """Stop the classifier."""
        self.logger.info(f"Stopping classifier '{self.name}'")
        self.state = BaseComponent.STATE_STOPPED
```

## Sample Implementation: RegimeDetector

A concrete example of a classifier is the RegimeDetector:

```python
# src/strategy/regime_detector.py
from typing import Any, Dict, Tuple, Optional
from .classifier import Classifier

class RegimeDetector(Classifier):
    """
    Detects market regimes based on configurable indicators and thresholds.
    """
    
    def __init__(self, 
                instance_name: str, 
                config_loader, 
                event_bus,
                component_config_key: str = None):
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
        super().setup()
        self._setup_regime_indicators()
    
    def _setup_regime_indicators(self):
        """Initialize indicators used for regime detection."""
        indicator_configs = self.get_specific_config("indicators", {})
        
        for indicator_name, config in indicator_configs.items():
            indicator_type = config.get("type")
            params = config.get("parameters", {})
            
            if indicator_type == "volatility":
                from ..indicators.volatility import VolatilityIndicator
                self._regime_indicators[indicator_name] = VolatilityIndicator(**params)
            elif indicator_type == "trend_strength":
                from ..indicators.trend import TrendStrengthIndicator
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
            if indicator.ready
        }
        
        # Apply regime classification rules
        detected_regime = self._classify_regime(indicator_values)
        
        # Apply regime change stabilization
        final_regime = self._apply_stabilization(detected_regime)
        
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
    
    def _apply_stabilization(self, detected_regime: str) -> str:
        """
        Apply stabilization to prevent rapid regime switching.
        
        Args:
            detected_regime: The newly detected regime
            
        Returns:
            str: The stabilized regime
        """
        current_classification = self.get_current_classification() or "default"
        
        # If no change, reset pending and increment duration
        if detected_regime == current_classification:
            self._pending_regime = None
            self._pending_duration = 0
            self._current_regime_duration += 1
            return current_classification
        
        # If new regime detected
        if self._pending_regime is None or detected_regime != self._pending_regime:
            # Start tracking a new pending regime
            self._pending_regime = detected_regime
            self._pending_duration = 1
            return current_classification
        
        # Continue tracking same pending regime
        self._pending_duration += 1
        
        # If pending regime has been stable for sufficient duration, change the regime
        if self._pending_duration >= self._min_regime_duration:
            self._current_regime_duration = 0
            self._pending_regime = None
            self._pending_duration = 0
            return detected_regime
        
        # Not stable enough yet, maintain current regime
        return current_classification
    
    def get_regime_data(self) -> Dict[str, Any]:
        """
        Get additional data about the current regime.
        
        Returns:
            Dict containing regime metadata and indicator values
        """
        return {
            'regime': self.get_current_classification(),
            'duration': self._current_regime_duration,
            'indicators': {
                name: indicator.value 
                for name, indicator in self._regime_indicators.items()
                if indicator.ready
            }
        }
```

## Usage in Optimization

The Classifier abstraction enables clean integration with the optimization framework:

```python
# In optimizer
def run_regime_optimization(self, strategy, data_handler, regime_detector):
    """
    Run optimization segmented by regime.
    
    Args:
        strategy: Strategy to optimize
        data_handler: Data handler for backtesting
        regime_detector: Regime classifier instance
    """
    # Run backtest with regime detection
    results = self._run_backtest_with_regime_tracking(strategy, data_handler, regime_detector)
    
    # Group trades by regime
    trades_by_regime = self._group_trades_by_regime(results['trades'])
    
    # Optimize parameters for each regime
    optimal_params_by_regime = {}
    for regime, trades in trades_by_regime.items():
        # Skip regimes with insufficient data
        if len(trades) < self._min_regime_samples:
            self.logger.warning(f"Insufficient samples for regime {regime}: {len(trades)} trades")
            continue
            
        # Calculate performance metrics for this regime
        regime_metrics = self._calculate_performance_metrics(trades)
        
        # Run optimization for this regime
        regime_params = self._optimize_for_regime(strategy, trades, regime_metrics)
        optimal_params_by_regime[regime] = regime_params
    
    return optimal_params_by_regime
```

## Relationship with Strategies

Classifiers can be used by strategies for either parameter selection or signal filtering:

```python
# In a regime-aware strategy
def on_bar(self, event):
    """Process bar event with regime awareness."""
    # Get market data
    data = event.get_data()
    
    # Get current regime
    regime = self._regime_detector.get_current_classification()
    
    # Option 1: Parameter Selection - Switch parameters based on regime
    if regime in self._regime_parameters:
        self.set_parameters(self._regime_parameters[regime])
    
    # Option 2: Signal Filtering - Adjust or filter signals based on regime
    signal = self._generate_signal(data)
    if signal:
        # Apply regime-based adjustments
        signal = self._adjust_signal_for_regime(signal, regime)
        if signal:  # Signal may be filtered out
            self.emit_signal(**signal)
```

## Benefits of the Classifier Abstraction

1. **Clean Separation**: Classifiers focus solely on identifying market conditions
2. **Reusability**: The same classifier can be used by strategies and optimization components
3. **Extensibility**: New classifier types can be easily created (volatility, trend, seasonality)
4. **Consistent Interface**: All classifiers follow the same usage pattern
5. **Event-Based Communication**: Classification changes trigger events for interested components
6. **Future Expansion**: Framework supports future enhancements like ML-based classifiers

## Future Development

The Classifier framework can be extended in several ways:

1. **Machine Learning Integration**: Support for ML-based classifiers
2. **Ensemble Classifiers**: Combine multiple classifiers for more robust regime detection
3. **Time-Varying Classification**: Handle time-of-day or seasonal effects
4. **Hierarchical Classification**: Support for nested classification (market state → volatility state)
5. **Classification Metrics**: Tools for evaluating classifier accuracy and stability

## Conclusion

The Classifier abstraction provides a clean architectural solution for regime detection and other market classification needs. By establishing classifiers as first-class components in the ADMF-Trader system, we enable more sophisticated analytics, optimization, and trading strategies without compromising the system's clean component-based design.