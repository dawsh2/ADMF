# Signal Processing in Risk Management

## Overview

This document outlines the design and implementation of signal processing capabilities within the ADMF-Trader risk module. Signal processing represents an extension of risk management that focuses on evaluating, filtering, and enhancing strategy-generated signals before they are translated into orders.

## Motivation

Trading strategies produce signals based on their specific logic, but these raw signals often benefit from additional processing to:

1. Reduce false positives and noise
2. Quantify signal confidence and quality
3. Apply meta-labeling from historical performance
4. Filter signals based on broader risk context
5. Smooth signals to prevent excessive trading

By implementing these capabilities within the risk module, we maintain a clean architectural flow:
* **Strategy module**: Consumes bar events, produces raw signals
* **Risk module**: Evaluates signals, applies signal processing, produces orders
* **Execution module**: Consumes orders, produces fills

## Architectural Design

Signal processing within the risk module follows these design principles:

1. **Layered Processing**: Signals pass through multiple processing layers before becoming orders
2. **Composable Components**: Individual processors can be combined in different configurations
3. **Stateful Analysis**: Processors can maintain state to track signal patterns over time
4. **Configuration-Driven**: Processing behavior is configurable rather than hard-coded
5. **Performance Attribution**: Each processor tracks its impact for analysis

## Architectural Placement of Regime Filtering

A key architectural decision is placing regime-based filtering in the Risk module rather than having strategies self-filter:

### Rationale

1. **Separation of Concerns**:
   - **Classifier Components** (like RegimeDetector) identify market conditions
   - **Strategy Components** focus on generating "pure" signals based on their core logic
   - **Risk Module** applies centralized filtering policies, including regime-based appropriateness

2. **Risk as the "Big Brain"**:
   - Risk module serves as the central authority for determining which signals become orders
   - Maintains consistent system-wide policies for trading appropriateness
   - Consolidates all filtering logic in one place for better governance

3. **Flexibility and Configuration**:
   - Regime filtering can be toggled or adjusted without modifying strategy code
   - Different filtering rules can be applied to different strategies
   - Rules can evolve based on performance without requiring strategy changes

### Implementation Approach

1. **Independent Classifiers**:
   - RegimeDetector and other classifiers are registered as independent components
   - They consume market data and maintain their classification state
   - They expose methods like `get_current_classification()` for other components

2. **"Pure" Strategy Signals**:
   - Strategies focus on their specific alpha-generation logic
   - Signals represent the strategy's "true intention" based on its indicators/rules
   - No regime-based filtering is applied at the strategy level

3. **Signal Processor with Regime Filter**:
   - The SignalProcessingPipeline includes a RegimeFilter processor
   - RegimeFilter queries the appropriate RegimeDetector component
   - It applies configured rules to filter signals based on regime/signal compatibility

This approach gives us the best of both worlds - strategies that focus on their core logic, and a risk system that ensures signals align with the broader market context before execution.

## Core Components

### SignalProcessor Base

```python
from abc import abstractmethod
from typing import Dict, Any, Optional, List
from ..core.component import BaseComponent

class SignalProcessor(BaseComponent):
    """
    Base class for components that process and evaluate strategy signals.
    
    Signal processors sit between raw strategy signals and order generation,
    providing filtering, enrichment, and transformation of signals.
    """
    
    def __init__(self, 
                instance_name: str, 
                config_loader,
                event_bus,
                component_config_key: str = None):
        super().__init__(instance_name, config_loader, event_bus, component_config_key)
        self._event_bus = event_bus
        
        # Statistics tracking
        self._processed_count = 0
        self._filtered_count = 0
        self._enhanced_count = 0
        
    @abstractmethod
    def process_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a trading signal.
        
        Args:
            signal: Raw signal from strategy
            
        Returns:
            Processed signal or None if signal should be filtered out
        """
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processor statistics.
        
        Returns:
            Dictionary of statistics about processing activity
        """
        return {
            'processed_count': self._processed_count,
            'filtered_count': self._filtered_count,
            'enhanced_count': self._enhanced_count,
            'filter_rate': self._filtered_count / max(1, self._processed_count),
            'enhancement_rate': self._enhanced_count / max(1, self._processed_count)
        }
```

### Signal Processing Pipeline

```python
class SignalProcessingPipeline(SignalProcessor):
    """
    Pipeline that applies multiple signal processors in sequence.
    """
    
    def __init__(self, 
                instance_name: str, 
                config_loader,
                event_bus,
                component_config_key: str = None):
        super().__init__(instance_name, config_loader, event_bus, component_config_key)
        self._processors = []
        
    def setup(self):
        """Set up processors from configuration."""
        self.state = BaseComponent.STATE_INITIALIZED
        
        # Load processor configurations
        processor_configs = self.get_specific_config("processors", [])
        
        # Create and add processors
        for config in processor_configs:
            processor_type = config.get("type")
            processor_name = config.get("name", f"{self.name}_{processor_type}")
            processor_config_key = config.get("config_key")
            
            # Create processor instance
            processor = self._create_processor(
                processor_type, 
                processor_name, 
                processor_config_key
            )
            
            if processor:
                self._processors.append(processor)
                processor.setup()
    
    def _create_processor(self, processor_type, name, config_key):
        """Create processor instance based on type."""
        # This would use a factory pattern or container to create instances
        pass
        
    def process_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process signal through the pipeline.
        
        Args:
            signal: Raw signal from strategy
            
        Returns:
            Processed signal or None if filtered out
        """
        self._processed_count += 1
        
        # Pass through each processor in sequence
        current_signal = signal
        
        for processor in self._processors:
            if current_signal is None:
                break
                
            current_signal = processor.process_signal(current_signal)
            
        if current_signal is None:
            self._filtered_count += 1
        elif current_signal != signal:
            self._enhanced_count += 1
            
        return current_signal
    
    def start(self):
        """Start all processors in the pipeline."""
        super().start()
        for processor in self._processors:
            processor.start()
            
    def stop(self):
        """Stop all processors in the pipeline."""
        for processor in self._processors:
            processor.stop()
        super().stop()
```

## Example Implementations

### Regime Filter

```python
class RegimeFilter(SignalProcessor):
    """
    Filters signals based on compatibility with the current market regime.
    
    This processor queries an independent Classifier component (typically a
    RegimeDetector) and vetoes signals that contradict the current regime
    according to configured rules.
    """
    
    def __init__(self, 
                instance_name: str, 
                config_loader,
                event_bus,
                component_config_key: str = None):
        super().__init__(instance_name, config_loader, event_bus, component_config_key)
        
        # Configuration
        self._regime_detector_key = self.get_specific_config("regime_detector_key", "regime_detector")
        self._allowed_regimes = self.get_specific_config("allowed_regimes", [])
        self._disallowed_regimes = self.get_specific_config("disallowed_regimes", [])
        self._bullish_regimes = self.get_specific_config("bullish_regimes", [])
        self._bearish_regimes = self.get_specific_config("bearish_regimes", [])
        self._neutral_regimes = self.get_specific_config("neutral_regimes", [])
        
        # Directional configuration - which regimes to veto for specific signal directions
        self._veto_long_in_regimes = self.get_specific_config("veto_long_in_regimes", [])
        self._veto_short_in_regimes = self.get_specific_config("veto_short_in_regimes", [])
        
        # Component references
        self._regime_detector = None
        self._container = None
        
    def setup(self):
        """Connect to the regime detector component."""
        self._container = self._get_container()
        
        if self._container:
            self._regime_detector = self._container.resolve(self._regime_detector_key)
            
            if self._regime_detector:
                self.state = BaseComponent.STATE_INITIALIZED
                self.logger.info(f"Connected to regime detector: {self._regime_detector_key}")
            else:
                self.logger.error(f"Failed to resolve regime detector: {self._regime_detector_key}")
                self.state = BaseComponent.STATE_FAILED
        else:
            self.logger.error("No container available to resolve regime detector")
            self.state = BaseComponent.STATE_FAILED
    
    def process_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process signal by applying regime-based filtering.
        
        Args:
            signal: Raw signal from strategy
            
        Returns:
            Signal or None if vetoed by regime rules
        """
        self._processed_count += 1
        
        # Get current regime from classifier
        if not self._regime_detector:
            # No regime detector available, pass signal through
            return signal
            
        current_regime = self._regime_detector.get_current_classification()
        if not current_regime:
            # No regime classification available, pass signal through
            return signal
            
        # Extract signal direction
        direction = signal.get('direction', '').upper()
        if not direction:
            # No direction, pass signal through
            return signal
            
        # Check if trading is allowed in this regime at all
        if self._allowed_regimes and current_regime not in self._allowed_regimes:
            self.logger.info(f"Signal vetoed: regime {current_regime} not in allowed_regimes")
            self._filtered_count += 1
            return None
            
        if current_regime in self._disallowed_regimes:
            self.logger.info(f"Signal vetoed: regime {current_regime} in disallowed_regimes")
            self._filtered_count += 1
            return None
            
        # Check direction-specific vetoes
        if direction == 'BUY' and current_regime in self._veto_long_in_regimes:
            self.logger.info(f"Long signal vetoed in regime: {current_regime}")
            self._filtered_count += 1
            return None
            
        if direction == 'SELL' and current_regime in self._veto_short_in_regimes:
            self.logger.info(f"Short signal vetoed in regime: {current_regime}")
            self._filtered_count += 1
            return None
            
        # Check bullish/bearish regime compatibility
        if direction == 'BUY' and current_regime in self._bearish_regimes:
            self.logger.info(f"Long signal vetoed in bearish regime: {current_regime}")
            self._filtered_count += 1
            return None
            
        if direction == 'SELL' and current_regime in self._bullish_regimes:
            self.logger.info(f"Short signal vetoed in bullish regime: {current_regime}")
            self._filtered_count += 1
            return None
            
        # Signal passed regime filtering
        # Add regime info to signal for downstream processors
        signal['regime'] = current_regime
        
        self._enhanced_count += 1
        return signal
        
    def _get_container(self):
        """Get the dependency injection container."""
        # Implementation depends on how your container is accessed
        pass
```

### Confidence-Based Signal Filter

```python
class ConfidenceFilter(SignalProcessor):
    """
    Filters signals based on confidence scores from meta-labeling.
    """
    
    def __init__(self, 
                instance_name: str, 
                config_loader,
                event_bus,
                component_config_key: str = None):
        super().__init__(instance_name, config_loader, event_bus, component_config_key)
        
        # Configuration
        self._min_confidence = self.get_specific_config("min_confidence", 0.6)
        self._confidence_model_path = self.get_specific_config("confidence_model_path")
        self._model = None
        self._feature_columns = []
        
    def setup(self):
        """Load and initialize the confidence model."""
        import joblib
        
        try:
            # Load model
            self._model = joblib.load(self._confidence_model_path)
            
            # Get feature columns from model metadata or config
            self._feature_columns = self.get_specific_config("feature_columns", [])
            
            self.state = BaseComponent.STATE_INITIALIZED
            self.logger.info(f"Confidence model loaded from {self._confidence_model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load confidence model: {e}")
            self.state = BaseComponent.STATE_FAILED
    
    def process_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process signal by applying confidence filtering.
        
        Args:
            signal: Raw signal from strategy
            
        Returns:
            Signal with confidence score or None if below threshold
        """
        self._processed_count += 1
        
        if self._model is None:
            return signal  # Pass through if model not loaded
            
        # Extract features from signal and market context
        features = self._extract_features(signal)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(features)
        
        # Add confidence to signal
        signal['confidence'] = confidence
        
        # Filter based on confidence threshold
        if confidence < self._min_confidence:
            self._filtered_count += 1
            return None
            
        self._enhanced_count += 1
        return signal
    
    def _extract_features(self, signal):
        """Extract model features from signal."""
        # Implementation depends on feature engineering approach
        pass
        
    def _calculate_confidence(self, features):
        """Calculate confidence score using the model."""
        # Implementation depends on model type
        pass
```

### Signal Smoothing Processor

```python
class SignalSmoother(SignalProcessor):
    """
    Smooths signals to reduce noise and prevent excessive trading.
    """
    
    def __init__(self, 
                instance_name: str, 
                config_loader,
                event_bus,
                component_config_key: str = None):
        super().__init__(instance_name, config_loader, event_bus, component_config_key)
        
        # Configuration
        self._smoothing_window = self.get_specific_config("smoothing_window", 3)
        self._min_consensus = self.get_specific_config("min_consensus", 0.66)
        
        # State
        self._signal_history = {}  # symbol -> list of signals
    
    def process_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process signal by applying smoothing.
        
        Args:
            signal: Raw signal from strategy
            
        Returns:
            Smoothed signal or None if filtered by smoothing
        """
        self._processed_count += 1
        
        symbol = signal.get('symbol')
        direction = signal.get('direction')
        
        if not symbol or not direction:
            return signal  # Pass through incomplete signals
            
        # Initialize history for symbol if needed
        if symbol not in self._signal_history:
            self._signal_history[symbol] = []
            
        # Add current signal to history
        self._signal_history[symbol].append(direction)
        
        # Trim history to window size
        self._signal_history[symbol] = self._signal_history[symbol][-self._smoothing_window:]
        
        # Check for consensus
        if len(self._signal_history[symbol]) < self._smoothing_window:
            # Not enough history yet
            self._filtered_count += 1
            return None
            
        # Count directions in window
        buy_count = self._signal_history[symbol].count('BUY')
        sell_count = self._signal_history[symbol].count('SELL')
        
        # Calculate consensus percentages
        buy_pct = buy_count / self._smoothing_window
        sell_pct = sell_count / self._smoothing_window
        
        # Check if signal matches the consensus direction
        if direction == 'BUY' and buy_pct >= self._min_consensus:
            self._enhanced_count += 1
            return signal
        elif direction == 'SELL' and sell_pct >= self._min_consensus:
            self._enhanced_count += 1
            return signal
        else:
            # Signal doesn't match consensus - filter it out
            self._filtered_count += 1
            return None
```

## Integration with Risk Management

The signal processing capabilities integrate with the existing risk management system:

```python
# Enhanced risk manager with signal processing
class EnhancedRiskManager(RiskManagerBase):
    """
    Risk manager with integrated signal processing capabilities.
    """
    
    def __init__(self, 
                instance_name: str, 
                config_loader,
                event_bus,
                component_config_key: str = None):
        super().__init__(instance_name, config_loader, event_bus, component_config_key)
        
        # Signal processor pipeline
        self._signal_processor = None
        self._processor_config_key = self.get_specific_config("signal_processor_config", None)
    
    def setup(self):
        """Set up risk manager and signal processor."""
        super().setup()
        
        # Create signal processor pipeline if configured
        if self._processor_config_key:
            self._signal_processor = SignalProcessingPipeline(
                f"{self.name}_signal_processor",
                self._config_loader,
                self._event_bus,
                self._processor_config_key
            )
            self._signal_processor.setup()
    
    def on_signal(self, event):
        """
        Process signal event.
        
        Args:
            event: Signal event data
        """
        signal_data = event.get_data()
        
        # Process signal through pipeline if available
        if self._signal_processor:
            processed_signal = self._signal_processor.process_signal(signal_data)
            
            # If signal was filtered out, stop processing
            if processed_signal is None:
                return
                
            # Use processed signal instead of original
            signal_data = processed_signal
        
        # Continue with normal risk management process
        position_size = self.size_position(signal_data)
        
        if position_size > 0:
            # Create and emit order
            order_data = self._create_order(signal_data, position_size)
            
            if self.validate_order(order_data):
                self.emit_order(order_data)
```

## Advanced Concepts

### Classifier Integration and Component Architecture

The integration between Classifiers (like RegimeDetector) and the Risk module follows these principles:

1. **Independence and Reusability**:
   - Classifiers are standalone components registered in the dependency injection container
   - They update their state based on market data events
   - Multiple consumers can query the same classifier

2. **Clean Information Flow**:
   - Classifier receives market data → updates its classification state
   - Strategy generates raw signals based on its logic
   - Risk module's RegimeFilter queries the classifier → applies filtering rules

3. **Explicit Dependency Management**:
   - RegimeFilter is configured with a `regime_detector_key` to specify which classifier to use
   - It resolves this dependency at setup time via the container
   - Multiple RegimeFilters can reference different classifiers (e.g., volatility, trend, etc.)

This approach offers a clean architectural solution where:
- Classifiers belong conceptually to the "strategy intelligence" domain
- Signal processing and filtering belong to the risk management domain
- Components are loosely coupled but explicitly connected via configuration

### Meta-Labeling Framework

Meta-labeling is a powerful technique that uses machine learning to assess whether a strategy's signals are likely to be profitable. The process involves:

1. Training a model on historical signals and their outcomes
2. Using that model to assign confidence scores to new signals
3. Filtering or sizing signals based on confidence

Implementation approach:

```python
class MetaLabelTrainer(BaseComponent):
    """
    Component for training meta-labeling models from historical trade data.
    """
    
    def train_model(self, strategy_name, lookback_days=365):
        """
        Train a meta-labeling model for a specific strategy.
        
        Args:
            strategy_name: Name of strategy to train for
            lookback_days: Historical data period to use
            
        Returns:
            Trained model
        """
        # Get historical trades and signals
        trades = self._get_historical_trades(strategy_name, lookback_days)
        
        # Create features from market data and signal properties
        features, labels = self._prepare_training_data(trades)
        
        # Train model
        model = self._train_model(features, labels)
        
        # Save model
        self._save_model(model, strategy_name)
        
        return model
```

### Signal Quality Framework

Beyond binary filtering, a comprehensive signal quality framework assesses multiple dimensions:

```python
class SignalQualityAnalyzer(SignalProcessor):
    """
    Analyzes signals across multiple quality dimensions.
    """
    
    def process_signal(self, signal):
        """
        Process signal by analyzing its quality.
        
        Args:
            signal: Raw trading signal
            
        Returns:
            Signal enhanced with quality metrics
        """
        # Analyze various quality dimensions
        quality_metrics = {
            'strength': self._calculate_strength(signal),
            'consistency': self._calculate_consistency(signal),
            'timeliness': self._calculate_timeliness(signal),
            'alignment': self._calculate_market_alignment(signal),
            'anomaly_score': self._calculate_anomaly_score(signal)
        }
        
        # Calculate composite quality score
        composite_score = self._calculate_composite_score(quality_metrics)
        
        # Add metrics to signal
        signal['quality'] = quality_metrics
        signal['quality_score'] = composite_score
        
        return signal
```

## Configuration Example

```yaml
# config.yaml
risk_manager:
  # Standard risk parameters
  position_sizing:
    default_size: 100
    max_position_pct: 0.05
    
  # Signal processing configuration
  signal_processor_config: signal_processors
  
# Signal processor configuration  
signal_processors:
  processors:
    - type: regime_filter
      name: directional_regime_filter
      config_key: regime_filters.directional
      
    - type: regime_filter
      name: volatility_regime_filter
      config_key: regime_filters.volatility
      
    - type: confidence_filter
      name: ml_confidence_filter
      config_key: confidence_models.meta_label
      
    - type: signal_smoother
      name: trend_smoother
      config_key: signal_smoothers.trend
      
# Individual processor configurations
regime_filters:
  directional:
    # Reference to the regime detector component
    regime_detector_key: "trend_regime_detector"
    
    # Directional regime configuration
    bullish_regimes: ["strong_uptrend", "weak_uptrend"]
    bearish_regimes: ["strong_downtrend", "weak_downtrend"]
    neutral_regimes: ["sideways", "consolidation"]
    
    # Explicit veto configuration (alternative to bullish/bearish)
    veto_long_in_regimes: ["strong_downtrend"]
    veto_short_in_regimes: ["strong_uptrend"]
    
  volatility:
    regime_detector_key: "volatility_regime_detector"
    allowed_regimes: ["medium_volatility", "low_volatility"]
    disallowed_regimes: ["extreme_volatility"]
    
confidence_models:
  meta_label:
    min_confidence: 0.65
    confidence_model_path: "models/meta_label_xgboost.pkl"
    feature_columns: ["price_momentum", "volume_ratio", "volatility", "signal_strength"]
    
signal_smoothers:
  trend:
    smoothing_window: 3
    min_consensus: 0.66
```

## Future Development

The signal processing framework can be extended in several directions:

1. **Advanced ML Integration**: Deep learning for signal quality assessment
2. **Adaptive Processing**: Dynamically adjusting processors based on market conditions
3. **Context-Aware Processing**: Incorporating broader market context into signal evaluation
4. **Online Learning**: Continuously updating models based on recent performance
5. **Processing Analytics**: Tools for visualizing and analyzing processor impact
6. **Regime-Specific Position Sizing**: Instead of binary veto, adjust position size based on regime compatibility

## Conclusion

Integrating signal processing into the risk module creates a powerful framework for enhancing trading signal quality. By filtering, transforming, and enriching strategy-generated signals before order creation, we can significantly improve system performance while maintaining architectural clarity.

The architectural decision to place regime filtering in the Risk module rather than at the Strategy level provides several key benefits:
1. Strategies remain focused on their core alpha generation
2. Risk policies are centralized and consistently applied
3. Filtering rules can evolve without requiring strategy modifications
4. The system maintains clear separation of concerns

This approach recognizes that risk management's domain extends beyond simple position sizing to include the fundamental question of signal quality and execution confidence - making it the true "big brain" of the trading system.