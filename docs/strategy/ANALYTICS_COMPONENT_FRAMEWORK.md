# Analytics Component Framework

## Motivation

In algorithmic trading systems, there's a critical need for components that analyze market conditions and trading signals without directly making trading decisions. The Analytics Component Framework addresses this architectural gap by providing a dedicated home for analytical capabilities that:

1. **Observe Without Deciding**: Analyze market data and signals without generating trading signals or orders
2. **Provide Context**: Supply contextual information to both Strategy and Risk modules
3. **Maintain Historical Analysis**: Track market conditions and signal performance over time
4. **Enable Feedback Loops**: Create formal pathways for system improvement based on analysis

### The Architectural Gap

Traditional trading architectures often scatter analytical functions across components:

- Embedding analysis in strategies reduces reusability and mixes concerns
- Placing analysis in risk components combines assessment with decision-making
- Implementing as utilities lacks proper lifecycle management and state tracking

This creates several problems:

- **Fragmented Analysis**: Similar analytics implemented inconsistently
- **Information Silos**: Analysis insights not shared across the system
- **Weak Feedback Loops**: No clear path for insights to improve system behavior
- **Limited Meta-Analysis**: Difficulty implementing sophisticated cross-cutting analysis

## Architectural Design

The Analytics Component Framework introduces a new category of components in the ADMF-Trader system that serves as a bridge between data sources and decision-making components:

```
                    Market Data
                         │
              ┌──────────┼──────────┐
              │          │          │
         ┌────▼────┐ ┌───▼────┐ ┌───▼────┐
         │Analytics│ │Strategy│ │  Risk  │
         │Component│ │ Module │ │ Module │
         └────┬────┘ └───┬────┘ └───┬────┘
              │          │          │
              └──────────┼──────────┘
                         │
                      Orders
```

### Design Principles

The Analytics Component Framework follows these key principles:

1. **Component-Based**: Consistent with ADMF-Trader's component architecture
2. **Explicit Interfaces**: Clear contracts with Strategy and Risk modules
3. **State Maintenance**: Proper lifecycle and state management
4. **Passive Observation**: Analysis without direct action
5. **Event-Driven**: Updates analysis based on system events
6. **Historical Tracking**: Maintains complete history with timestamps
7. **Clean Integration**: Works seamlessly with existing components

## Directory Structure

The Analytics Component Framework lives in the Strategy module, consistent with its role as an "upstream" component that provides insights to other parts of the system:

```
src/strategy/
├── analytics/
│   ├── __init__.py
│   ├── base.py              # The base AnalyticsComponent
│   ├── classifier.py        # Classifier abstract class
│   ├── meta_labeler.py      # MetaLabeler abstract class
│   ├── classifiers/
│   │   ├── __init__.py
│   │   ├── regime_detector.py
│   │   └── volatility_classifier.py
│   └── meta_labelers/
│       ├── __init__.py
│       ├── signal_quality_labeler.py
│       └── regime_performance_labeler.py
└── ... (other strategy files)
```

## Component Types

The framework defines two primary types of analytics components:

1. **Classifiers**: Components that analyze market data and produce categorical labels
2. **MetaLabelers**: Components that analyze signals/decisions and evaluate their quality

### Key Differences

| Feature | Classifier | MetaLabeler |
|---------|------------|-------------|
| **Input** | Market data (bars) | Trading signals |
| **Output** | Market state labels | Signal quality assessments |
| **Source Events** | BAR events | SIGNAL and FILL events |
| **Time Focus** | Current market state | Historical performance patterns |
| **Examples** | RegimeDetector | SignalQualityLabeler |

## Implementation Approach

### AnalyticsComponent Base Class

The framework is built around a new abstract base class that extends the existing BaseComponent:

```python
from abc import abstractmethod
from typing import Dict, Any, List, Optional
from ...core.component import BaseComponent

class AnalyticsComponent(BaseComponent):
    """
    Base class for components that analyze data without making trading decisions.
    
    Analytics components maintain state, provide context to other components,
    and offer insights without directly generating or filtering signals.
    """
    
    def __init__(self, 
                instance_name: str, 
                config_loader, 
                event_bus,
                component_config_key: str = None):
        """Initialize the analytics component."""
        super().__init__(instance_name, config_loader, component_config_key)
        self._event_bus = event_bus
        self._history = []
        self._current_analysis = {}
    
    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data and produce results.
        
        Args:
            data: Data to analyze
            
        Returns:
            Analysis results
        """
        pass
    
    def get_current_analysis(self) -> Dict[str, Any]:
        """
        Get the current analysis results.
        
        Returns:
            Current analysis results
        """
        return self._current_analysis
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get analysis history.
        
        Returns:
            List of historical analysis records
        """
        return self._history
        
    def setup(self):
        """Set up analytics component resources."""
        self.logger.info(f"Setting up analytics component '{self.name}'")
        # Subscribe to relevant events based on component type
        if self._event_bus:
            self._subscribe_to_events()
        self.state = BaseComponent.STATE_INITIALIZED
        
    def _subscribe_to_events(self):
        """Subscribe to relevant events - to be implemented by subclasses."""
        pass
        
    def start(self):
        """Start the analytics component."""
        self.logger.info(f"Starting analytics component '{self.name}'")
        self.state = BaseComponent.STATE_STARTED
    
    def stop(self):
        """Stop the analytics component."""
        self.logger.info(f"Stopping analytics component '{self.name}'")
        if self._event_bus:
            try:
                self._unsubscribe_from_events()
            except Exception as e:
                self.logger.error(f"Error unsubscribing from events: {e}")
        self.state = BaseComponent.STATE_STOPPED
        
    def _unsubscribe_from_events(self):
        """Unsubscribe from events - to be implemented by subclasses."""
        pass
```

### Classifier Component

**Event Broadcasting Design**: Classifiers implement a change-based event broadcasting pattern. Events are only emitted when the classification changes, not on every bar. This design decision:
- Dramatically reduces event traffic and logging noise
- Maintains full data availability (current state is always queryable)
- Enables efficient historical reconstruction from change events
- Follows reactive programming best practices

```python
from .base import AnalyticsComponent
from typing import Dict, Any, Optional, List
from ...core.event import Event, EventType

class Classifier(AnalyticsComponent):
    """
    Base class for market condition classifiers.
    
    Classifiers analyze market data and produce categorical labels without
    directly generating trading signals. Examples include regime detectors,
    market state classifiers, and volatility classifiers.
    """
    
    def __init__(self, 
                instance_name: str, 
                config_loader, 
                event_bus,
                component_config_key: str = None):
        """Initialize the Classifier."""
        super().__init__(instance_name, config_loader, event_bus, component_config_key)
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
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement analyze method from AnalyticsComponent.
        
        Delegates to classify and updates state.
        
        Args:
            data: Market data to analyze
            
        Returns:
            Dict with classification results
        """
        # Get classification from concrete implementation
        classification = self.classify(data)
        
        # Check if classification changed
        classification_changed = classification != self._current_classification
        
        # Update current classification
        self._current_classification = classification
        
        # Record in history
        record = {
            'timestamp': data.get('timestamp'),
            'classification': classification,
            'changed': classification_changed
        }
        self._classification_history.append(record)
        self._history.append(record)
        
        # Update current results
        self._current_analysis = {
            'classification': classification,
            'changed': classification_changed,
            'timestamp': data.get('timestamp')
        }
        
        # Emit classification event if changed
        if classification_changed and self._event_bus:
            self._event_bus.publish(self._create_classification_event(data, classification))
            
        return self._current_analysis
    
    def get_current_classification(self) -> Optional[str]:
        """
        Returns the current classification label.
        
        Returns:
            str: Current classification label or None if not yet classified
        """
        return self._current_classification
    
    def get_classification_history(self) -> List[Dict[str, Any]]:
        """
        Returns the history of classifications with timestamps.
        
        Returns:
            list: List of classification events with timestamps
        """
        return self._classification_history
        
    def _subscribe_to_events(self):
        """Subscribe to BAR events."""
        self._event_bus.subscribe(EventType.BAR, self.on_bar)
    
    def _unsubscribe_from_events(self):
        """Unsubscribe from BAR events."""
        self._event_bus.unsubscribe(EventType.BAR, self.on_bar)
    
    def on_bar(self, event: Event):
        """
        Process bar event and update classification.
        
        Args:
            event: Bar event with market data
        """
        # Extract market data from event
        data = event.payload
        
        # Analyze the data (which delegates to classify)
        self.analyze(data)
    
    def _create_classification_event(self, data: Dict[str, Any], classification: str) -> Event:
        """
        Create a classification event.
        
        Args:
            data: Market data that triggered the classification
            classification: The classification label
            
        Returns:
            Event object
        """
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
```

### MetaLabeler Component

```python
from .base import AnalyticsComponent
from typing import Dict, Any, List, Optional
from ...core.event import Event, EventType
import uuid

class MetaLabeler(AnalyticsComponent):
    """
    Base class for signal quality meta-labelers.
    
    MetaLabelers add a second layer of analysis to evaluate trading signals without 
    generating new signals. They assess signal quality, predict outcomes, and
    provide feedback to improve strategy and risk decisions.
    """
    
    def __init__(self, 
                instance_name: str, 
                config_loader, 
                event_bus,
                component_config_key: str = None):
        """Initialize the MetaLabeler."""
        super().__init__(instance_name, config_loader, event_bus, component_config_key)
        self._signal_history = []
        self._signals_by_id = {}
        
    def _subscribe_to_events(self):
        """Subscribe to relevant events."""
        self._event_bus.subscribe(EventType.SIGNAL, self.on_signal)
        self._event_bus.subscribe(EventType.FILL, self.on_fill)
        self._event_bus.subscribe(EventType.TRADE_COMPLETE, self.on_trade_complete)
    
    def _unsubscribe_from_events(self):
        """Unsubscribe from events."""
        self._event_bus.unsubscribe(EventType.SIGNAL, self.on_signal)
        self._event_bus.unsubscribe(EventType.FILL, self.on_fill)
        self._event_bus.unsubscribe(EventType.TRADE_COMPLETE, self.on_trade_complete)
    
    def on_signal(self, event: Event):
        """Record signal for analysis."""
        signal_data = event.payload
        self._record_signal(signal_data)
        self._run_meta_labeling(signal_data)
        
    def on_fill(self, event: Event):
        """Update signal with fill information."""
        fill_data = event.payload
        self._update_signal_fill(fill_data)
        
    def on_trade_complete(self, event: Event):
        """Update signal with final outcome information."""
        trade_data = event.payload
        self._update_signal_outcome(trade_data)
        
        # Run analysis after trade completion to update models
        self.analyze(self._signal_history)
        
    @abstractmethod
    def meta_label(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate signal quality and add meta-labels.
        
        Args:
            signal: Signal to evaluate
            
        Returns:
            Signal with added meta-labels
        """
        pass
    
    def _run_meta_labeling(self, signal_data: Dict[str, Any]):
        """
        Run meta-labeling on a signal and publish results.
        
        Args:
            signal_data: Signal event data
        """
        # Skip if signal ID is missing
        if 'id' not in signal_data:
            return
            
        # Apply meta-labeling
        meta_labeled_signal = self.meta_label(signal_data)
        
        # Publish meta-label event
        if meta_labeled_signal and self._event_bus:
            meta_label_event = Event(
                EventType.META_LABEL,
                meta_labeled_signal
            )
            self._event_bus.publish(meta_label_event)
        
    def analyze(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze signals to derive insights and update internal models.
        
        Args:
            signals: List of signals to analyze
            
        Returns:
            Dict containing analysis results
        """
        # Skip if not enough data
        if not signals:
            return {}
            
        # Perform basic analysis
        total_signals = len(signals)
        completed_signals = sum(1 for s in signals if s.get('status') == 'completed')
        wins = sum(1 for s in signals if s.get('status') == 'completed' and s.get('outcome') == 'win')
        losses = sum(1 for s in signals if s.get('status') == 'completed' and s.get('outcome') == 'loss')
        
        win_rate = wins / completed_signals if completed_signals > 0 else 0
        
        analysis = {
            'timestamp': signals[-1].get('timestamp') if signals else None,
            'total_signals': total_signals,
            'completed_signals': completed_signals,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate
        }
        
        # Update state
        self._current_analysis = analysis
        self._history.append(analysis)
        
        return analysis
        
    def _record_signal(self, signal_data: Dict[str, Any]) -> None:
        """
        Record a signal in the analyzer history.
        
        Args:
            signal_data: Signal event data
        """
        signal_id = signal_data.get('id', str(uuid.uuid4()))
        
        # Create signal record
        signal_record = {
            'id': signal_id,
            'timestamp': signal_data.get('timestamp'),
            'strategy': signal_data.get('strategy'),
            'symbol': signal_data.get('symbol'),
            'direction': signal_data.get('direction'),
            'price': signal_data.get('price'),
            'quantity': signal_data.get('quantity'),
            'reason': signal_data.get('reason'),
            'status': 'generated',
            'fills': [],
            'trades': [],
            'outcome': None,
            'pnl': None
        }
        
        # Store the record
        self._signals_by_id[signal_id] = signal_record
        self._signal_history.append(signal_record)
        
    def _update_signal_fill(self, fill_data: Dict[str, Any]) -> None:
        """
        Update signal records with fill information.
        
        Args:
            fill_data: Fill event data
        """
        signal_id = fill_data.get('signal_id')
        if signal_id and signal_id in self._signals_by_id:
            signal_record = self._signals_by_id[signal_id]
            
            # Update status
            signal_record['status'] = 'filled'
            
            # Add fill to record
            signal_record['fills'].append({
                'fill_id': fill_data.get('id', str(uuid.uuid4())),
                'timestamp': fill_data.get('timestamp'),
                'price': fill_data.get('price'),
                'quantity': fill_data.get('quantity')
            })
            
    def _update_signal_outcome(self, trade_data: Dict[str, Any]) -> None:
        """
        Update signal records with final outcome information.
        
        Args:
            trade_data: Trade completion event data
        """
        signal_id = trade_data.get('signal_id')
        if signal_id and signal_id in self._signals_by_id:
            signal_record = self._signals_by_id[signal_id]
            
            # Update status
            signal_record['status'] = 'completed'
            
            # Add trade outcome
            signal_record['trades'].append({
                'trade_id': trade_data.get('id', str(uuid.uuid4())),
                'entry_time': trade_data.get('entry_time'),
                'exit_time': trade_data.get('exit_time'),
                'entry_price': trade_data.get('entry_price'),
                'exit_price': trade_data.get('exit_price'),
                'quantity': trade_data.get('quantity'),
                'pnl': trade_data.get('pnl'),
                'return_pct': trade_data.get('return_pct')
            })
            
            # Update overall signal outcome
            signal_record['outcome'] = 'win' if trade_data.get('pnl', 0) > 0 else 'loss'
            signal_record['pnl'] = trade_data.get('pnl')
            
    def get_signal_history(self) -> List[Dict[str, Any]]:
        """
        Get the recorded signal history.
        
        Returns:
            List of signal records with outcomes
        """
        return self._signal_history
        
    def get_insights(self) -> List[Dict[str, Any]]:
        """
        Get actionable insights from analysis.
        
        Returns:
            List of insight dictionaries
        """
        return []
```

## Implementation Examples

### RegimeDetector (Classifier Implementation)

```python
from ..analytics.classifier import Classifier
from typing import Dict, Any, Optional

class RegimeDetector(Classifier):
    """
    Detects market regimes based on configurable indicators and thresholds.
    Implements stabilization logic to prevent rapid regime switching.
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
            
            # Create indicators based on configuration
            if indicator_type == "rsi":
                from ...strategy.components.indicators.oscillators import RSIIndicator
                self._regime_indicators[indicator_name] = RSIIndicator(
                    instance_name=f"{self.name}_{indicator_name}",
                    config_loader=self._config_loader,
                    parameters=params
                )
            elif indicator_type == "atr":
                from ...strategy.components.indicators.volatility import ATRIndicator
                self._regime_indicators[indicator_name] = ATRIndicator(
                    instance_name=f"{self.name}_{indicator_name}",
                    config_loader=self._config_loader,
                    parameters=params
                )
    
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
        final_regime = self._apply_stabilization(detected_regime)
        
        return final_regime
    
    def _classify_regime(self, indicator_values: Dict[str, float]) -> str:
        """Apply rules to classify the current regime based on indicator values."""
        # Implementation details...
        pass
        
    def _apply_stabilization(self, detected_regime: str) -> str:
        """Apply stabilization to prevent rapid regime switching."""
        # Implementation details...
        pass
```

### SignalQualityLabeler (MetaLabeler Implementation)

```python
from ..analytics.meta_labeler import MetaLabeler
from typing import Dict, Any, List
import numpy as np

class SignalQualityLabeler(MetaLabeler):
    """
    Evaluates signal quality based on historical performance patterns.
    
    This meta-labeler analyzes past signals to predict the quality
    of new signals, adding confidence scores and probability estimates.
    """
    
    def __init__(self, 
                instance_name: str, 
                config_loader, 
                event_bus,
                component_config_key: str = None):
        super().__init__(instance_name, config_loader, event_bus, component_config_key)
        
        # Configuration
        self._min_signals = self.get_specific_config("min_signals_for_training", 30)
        self._confidence_threshold = self.get_specific_config("confidence_threshold", 0.6)
        self._regime_aware = self.get_specific_config("regime_aware", True)
        
        # State
        self._performance_by_feature = {}
        self._regime_detector = None
        
    def setup(self):
        """Set up meta-labeler."""
        super().setup()
        
        # Resolve regime detector if regime-aware
        if self._regime_aware:
            self._container = self._get_container()
            regime_key = self.get_specific_config("regime_detector_key", "regime_detector")
            self._regime_detector = self._container.resolve(regime_key)
    
    def meta_label(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate signal quality and add meta-labels.
        
        Args:
            signal: Signal to evaluate
            
        Returns:
            Signal with added meta-labels
        """
        # Skip if not enough data for reliable labeling
        if len(self._signal_history) < self._min_signals:
            return self._add_default_labels(signal)
            
        # Extract features from signal
        features = self._extract_features(signal)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence(features, signal)
        
        # Create enhanced signal with meta-labels
        enhanced_signal = dict(signal)  # Make a copy
        enhanced_signal['meta_labels'] = {
            'confidence': confidence_scores['overall_confidence'],
            'win_probability': confidence_scores['win_probability'],
            'expected_return': confidence_scores['expected_return'],
            'quality_score': confidence_scores['quality_score'],
            'recommendation': 'take' if confidence_scores['overall_confidence'] >= self._confidence_threshold else 'skip'
        }
        
        return enhanced_signal
    
    def _extract_features(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from signal for meta-labeling."""
        features = {}
        
        # Basic signal features
        features['direction'] = 1 if signal.get('direction') == 'BUY' else -1
        features['reason'] = signal.get('reason')
        features['strategy'] = signal.get('strategy')
        
        # Get regime if available
        if self._regime_detector:
            features['regime'] = self._regime_detector.get_current_classification()
        
        # Add more features based on market context, time of day, etc.
        
        return features
    
    def _calculate_confidence(self, features: Dict[str, Any], signal: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores based on features."""
        # Initialize results
        results = {
            'overall_confidence': 0.5,  # Default middle confidence
            'win_probability': 0.5,
            'expected_return': 0.0,
            'quality_score': 0.5
        }
        
        # Skip if not enough historical data
        if len(self._signal_history) < self._min_signals:
            return results
            
        # Find similar past signals
        similar_signals = self._find_similar_signals(features)
        
        # If no similar signals, return default
        if not similar_signals:
            return results
            
        # Calculate metrics from similar signals
        win_count = sum(1 for s in similar_signals if s.get('outcome') == 'win')
        total_count = len(similar_signals)
        
        if total_count > 0:
            win_rate = win_count / total_count
            
            # Calculate average return for wins and losses
            win_returns = [s.get('pnl', 0) for s in similar_signals if s.get('outcome') == 'win']
            loss_returns = [s.get('pnl', 0) for s in similar_signals if s.get('outcome') == 'loss']
            
            avg_win = np.mean(win_returns) if win_returns else 0
            avg_loss = np.mean(loss_returns) if loss_returns else 0
            
            # Calculate expected return
            expected_return = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
            
            # Calculate quality score (win rate adjusted by return)
            quality_score = win_rate * (1 + (expected_return / 100)) if expected_return > 0 else win_rate * 0.5
            
            # Update results
            results['win_probability'] = win_rate
            results['expected_return'] = expected_return
            results['quality_score'] = quality_score
            results['overall_confidence'] = quality_score
        
        return results
    
    def _find_similar_signals(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar signals in history based on features."""
        similar_signals = []
        
        for signal in self._signal_history:
            if signal.get('status') != 'completed':
                continue
                
            # Match key features
            if (signal.get('direction') == features.get('direction') and
                signal.get('strategy') == features.get('strategy')):
                
                # For regime-aware matching
                if self._regime_aware:
                    if signal.get('regime') == features.get('regime'):
                        similar_signals.append(signal)
                else:
                    similar_signals.append(signal)
        
        return similar_signals
    
    def _add_default_labels(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Add default meta-labels when not enough data is available."""
        enhanced_signal = dict(signal)  # Make a copy
        enhanced_signal['meta_labels'] = {
            'confidence': 0.5,
            'win_probability': 0.5,
            'expected_return': 0.0,
            'quality_score': 0.5,
            'recommendation': 'neutral'
        }
        return enhanced_signal
        
    def _get_container(self):
        """Get dependency container."""
        # Implementation depends on how container is accessed in your system
        pass
```

## Component Registration

```python
# In system initialization
container.register_type(
    'regime_detector', 
    RegimeDetector, 
    instance_name='market_regime_detector', 
    component_config_key='analytics.classifiers.regime_detector'
)
                   
container.register_type(
    'signal_quality_labeler', 
    SignalQualityLabeler, 
    instance_name='signal_quality_labeler', 
    component_config_key='analytics.meta_labelers.signal_quality'
)
```

## Configuration Example

```yaml
# In config.yaml
analytics:
  classifiers:
    regime_detector:
      indicators:
        volatility:
          type: "atr"
          parameters:
            period: 14
        rsi:
          type: "rsi"
          parameters:
            period: 14
            
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
            
      min_regime_duration: 5
      
  meta_labelers:
    signal_quality:
      min_signals_for_training: 30
      confidence_threshold: 0.6
      regime_aware: true
      regime_detector_key: "regime_detector"
```

## Usage in Strategy and Risk Components

### Strategy Integration

```python
class RegimeAwareStrategy(BaseComponent):
    """Strategy that adapts to market regimes based on analytics."""
    
    def setup(self):
        """Set up strategy and connect to analytics components."""
        super().setup()
        
        # Get analytics components from container
        self._container = self._get_container()
        self._regime_detector = self._container.resolve('regime_detector')
        
        # Set up event subscriptions
        self._event_bus.subscribe(EventType.BAR, self.on_bar)
        
    def on_bar(self, event):
        """Process bar event with regime awareness."""
        # Get market data
        data = event.payload
        
        # Skip irrelevant symbols
        if data.get('symbol') != self._symbol:
            return
        
        # Check current regime
        current_regime = None
        if self._regime_detector:
            current_regime = self._regime_detector.get_current_classification()
        
        # Apply regime-specific parameters if available
        if current_regime and current_regime in self._regime_parameters:
            self.set_parameters(self._regime_parameters[current_regime])
        
        # Generate signals with standard strategy logic
        signals = self.calculate_signals(data)
        
        # Annotate signals with regime
        for signal in signals:
            signal['regime'] = current_regime
            
            # Emit signal
            self.emit_signal(**signal)
```

### Risk Integration

```python
class AnalyticsAwareRiskManager(BaseComponent):
    """Risk manager that leverages analytics insights."""
    
    def setup(self):
        """Set up risk manager and connect to analytics components."""
        super().setup()
        
        # Get analytics components from container
        self._container = self._get_container()
        self._regime_detector = self._container.resolve('regime_detector')
        
        # Set up event subscriptions
        self._event_bus.subscribe(EventType.SIGNAL, self.on_signal)
        self._event_bus.subscribe(EventType.META_LABEL, self.on_meta_label)
        
    def on_signal(self, event):
        """Process signal, but wait for meta-labeling."""
        # Store signal for later processing after meta-labeling
        signal_data = event.payload
        self._pending_signals[signal_data.get('id')] = signal_data
        
    def on_meta_label(self, event):
        """Process signal with meta-label insights."""
        # Get enhanced signal with meta-labels
        enhanced_signal = event.payload
        signal_id = enhanced_signal.get('id')
        
        # Skip if we don't have the original signal
        if signal_id not in self._pending_signals:
            return
            
        # Remove from pending
        del self._pending_signals[signal_id]
        
        # Apply meta-label insights to risk decisions
        meta_labels = enhanced_signal.get('meta_labels', {})
        
        # Skip low-confidence signals
        if meta_labels.get('recommendation') == 'skip':
            self.logger.info(f"Skipping low-quality signal: {enhanced_signal}")
            return
            
        # Adjust position size based on signal quality
        quality_scalar = meta_labels.get('quality_score', 0.5)
        enhanced_signal['_quality_scalar'] = quality_scalar
        
        # Apply standard risk management with quality scaling
        order = self._create_order(enhanced_signal)
        if self._validate_order(order):
            self._emit_order(order)
```

## Benefits of the Analytics Component Framework

1. **Clean Architecture**: Separate analytics from decision-making
2. **Reusable Components**: Analytics can be used by both Strategy and Risk
3. **Historical Tracking**: Proper state history for analysis and improvement
4. **Formal Analysis**: Structured approach to market and signal analysis
5. **Flexible Framework**: Can be extended to cover many analytical needs
6. **Event Integration**: Seamless integration with existing event system
7. **Configuration-Driven**: Easy adaptation through configuration

## The Role of Meta-Labeling

The MetaLabeler component formalizes the concept of meta-labeling in trading systems, where a "second layer" of analysis evaluates the quality of trading signals. As introduced by Marcos López de Prado, meta-labeling:

1. Separates signal generation from decision-making
2. Uses historical outcomes to predict future success
3. Focuses on quality rather than just direction
4. Can optimize for multiple objectives

Unlike traditional meta-labeling that is often implemented as an ad-hoc process, this framework provides:

1. Proper component lifecycle management
2. Historical tracking of signals and outcomes
3. Integration with other analytics like regime detection
4. Clean architecture to separate concerns

## Future Development

The Analytics Component Framework can be extended in several directions:

1. **Machine Learning Integration**: Add ML-based classifiers and meta-labelers
2. **Hierarchical Analytics**: Complex analytics that combine multiple classifiers
3. **Ensemble Approaches**: Meta-labelers that combine multiple evaluation methods
4. **Adaptive Learning**: Components that adjust their analysis over time
5. **Performance Attribution**: More sophisticated attribution of performance to factors
6. **Visual Analytics**: Components that generate visualizations of analytics results

## Conclusion

The Analytics Component Framework fills a critical architectural gap in the ADMF-Trader system by providing a dedicated home for components that analyze market conditions and signal quality. By properly separating analytical responsibilities from trading decisions, we create a more modular, maintainable, and extensible system.

This framework enables sophisticated analytical capabilities while maintaining the clean component-based architecture of the existing system. The formal interfaces ensure consistent usage patterns and clear communication between analytics components and the rest of the system.

By implementing this framework, we lay the groundwork for more advanced regime-based optimization, signal quality assessment, and adaptive trading strategies, all without compromising architectural integrity.