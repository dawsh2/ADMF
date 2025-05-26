# Signal-Based Weight Optimization Implementation Plan

## Core Concept: Signals as Currency

Instead of running full backtests for weight optimization, replay **signal events** through the existing event-driven architecture. This maintains temporal integrity while providing massive efficiency gains.

## Architecture Overview

### Current Flow
```
Bars → Strategy → Signals → Portfolio → Performance
```

### Proposed Flow
```
Phase 1 (Parameter Optimization): Bars → Strategy → Signals (CAPTURED) → Portfolio → Performance
Phase 2 (Weight Optimization): Stored Signals (REPLAYED) → Portfolio → Performance  
```

## Key Components

### 1. SignalCollector Component
```python
class SignalCollector(BaseComponent):
    """Captures signals during parameter optimization with regime context"""
    
    def __init__(self):
        self.regime_signals = {}  # {regime: [signal_events]}
        
    def on_signal_event(self, event: Event):
        current_regime = self.get_current_regime()
        
        signal_event_data = {
            'timestamp': event.payload['timestamp'],
            'symbol': event.payload['symbol'],
            'signal_type': event.payload['signal_type'],
            'price_at_signal': event.payload['price_at_signal'],
            'ma_strength': event.payload.get('ma_strength', 0),
            'rsi_strength': event.payload.get('rsi_strength', 0),
            'regime': current_regime
        }
        
        if current_regime not in self.regime_signals:
            self.regime_signals[current_regime] = []
        self.regime_signals[current_regime].append(signal_event_data)
```

### 2. RegimeSignalReplayer Component
```python
class RegimeSignalReplayer(BaseComponent):
    """Replays regime-specific signals with new weights through event bus"""
    
    def replay_regime_signals(self, regime: str, signals: List[Dict], weights: Dict):
        # Set new weights on strategy
        strategy = self._container.resolve(self._strategy_service_name)
        strategy.set_parameters(weights)
        
        # Reset portfolio state
        portfolio = self._container.resolve(self._portfolio_service_name)
        portfolio.reset()
        
        # Replay signals in temporal order
        for signal_data in sorted(signals, key=lambda x: x['timestamp']):
            weighted_signal_event = self._create_weighted_signal_event(signal_data, weights)
            self._event_bus.publish(weighted_signal_event)
        
        return portfolio.get_performance()
```

### 3. Weight-Adjusted Signal Generation
```python
def _create_weighted_signal_event(self, original_signal: Dict, weights: Dict) -> Event:
    """Apply new weights to historical signal data and generate new signal event"""
    
    # Extract component signals
    ma_strength = original_signal.get('ma_strength', 0)
    rsi_strength = original_signal.get('rsi_strength', 0)
    
    # Apply new weights
    ma_weighted = ma_strength * weights['ma_rule.weight']
    rsi_weighted = rsi_strength * weights['rsi_rule.weight']
    combined_strength = ma_weighted + rsi_weighted
    
    # Determine new signal type
    if combined_strength > 0.6:
        new_signal_type = 1  # BUY
    elif combined_strength < -0.6:
        new_signal_type = -1  # SELL
    else:
        new_signal_type = 0  # HOLD
    
    # Create weighted signal event
    weighted_payload = {
        'timestamp': original_signal['timestamp'],
        'symbol': original_signal['symbol'],
        'signal_type': new_signal_type,
        'price_at_signal': original_signal['price_at_signal'],
        'signal_strength': abs(combined_strength),
        'strategy_id': 'weight_optimized_ensemble',
        'reason': f'Weighted({weights["ma_rule.weight"]:.2f}/{weights["rsi_rule.weight"]:.2f})'
    }
    
    return Event(EventType.SIGNAL, weighted_payload)
```

## Integration with Genetic Optimizer

### Modified Fitness Evaluation
```python
def _evaluate_fitness_on_regime_signals(self, individual: Dict, regime_signals: List[Dict]) -> float:
    """Fitness evaluation using signal replay instead of full backtest"""
    
    weights = {
        'ma_rule.weight': individual['ma_rule.weight'],
        'rsi_rule.weight': individual['rsi_rule.weight']
    }
    
    performance = self.signal_replayer.replay_regime_signals(
        regime=self.current_regime,
        signals=regime_signals, 
        weights=weights
    )
    
    return performance[self._metric_to_optimize]
```

## Implementation Benefits

### ✅ Maintains Event-Driven Architecture
- No lookahead bias
- Temporal integrity preserved
- Existing portfolio/execution logic unchanged

### ✅ Massive Efficiency Gains
- Only replay signals (not full bars)
- Regime-specific datasets (25-70% smaller)
- Parallel regime optimization possible

### ✅ Clean Integration
- Builds on existing signal infrastructure
- No parallel simulation engine needed
- Straightforward 6-8 hour implementation

## Implementation Steps

1. **Signal Collection Integration** (~2 hours)
   - Add SignalCollector to parameter optimization phase
   - Hook into existing signal events with regime context

2. **Signal Replayer Component** (~3 hours)
   - Build RegimeSignalReplayer component
   - Implement weight-adjusted signal generation

3. **Genetic Optimizer Modification** (~2 hours)
   - Add signal-based fitness evaluation mode
   - Switch between full backtest vs signal replay

4. **Testing & Validation** (~2 hours)
   - Verify signal-based results match full backtest results
   - Performance benchmarking

**Total Effort**: ~8-10 hours of focused implementation

## Data Flow Modes

The optimizers will need to support dual operation modes:

### Mode 1: Full Backtest (Current)
- **Input**: Rules/Parameters + Full Bar Dataset
- **Process**: Complete event-driven backtest
- **Use Case**: Parameter optimization, validation

### Mode 2: Signal Replay (Proposed)  
- **Input**: Signals/Classifiers + Weight Parameters
- **Process**: Regime-filtered signal replay
- **Use Case**: Weight optimization, regime-specific training

## Configuration

```yaml
weight_optimization:
  mode: "signal_replay"  # or "full_backtest"
  signal_storage_enabled: true
  regime_filtering: true
  parallel_regime_optimization: true
```

This approach makes signals the fundamental currency while preserving all architectural benefits of the event-driven system.