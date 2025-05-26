"""
Base classes for the ADMF-Trader strategy module.

This module provides the abstract base classes and core implementations
for building trading strategies within the ADMF framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..core.component import Component
from ..core.event import Event


class SignalType(Enum):
    """Types of trading signals."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"


@dataclass
class Signal:
    """Represents a trading signal."""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    strength: float  # -1.0 to 1.0
    rule_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate signal strength."""
        if not -1.0 <= self.strength <= 1.0:
            raise ValueError(f"Signal strength must be between -1.0 and 1.0, got {self.strength}")


@dataclass
class ParameterSchema:
    """Defines the schema for a parameter."""
    name: str
    type: type
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str = ""
    
    def validate(self, value: Any) -> bool:
        """Validate a parameter value against this schema."""
        if not isinstance(value, self.type):
            return False
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


class ParameterSet:
    """Manages strategy parameters with validation."""
    
    def __init__(self, schema: Dict[str, ParameterSchema]):
        self.schema = schema
        self._parameters: Dict[str, Any] = {}
        self._version: int = 0
        
        # Initialize with defaults
        for name, param_schema in schema.items():
            self._parameters[name] = param_schema.default
    
    def update(self, **kwargs) -> None:
        """Update parameters with validation."""
        for name, value in kwargs.items():
            if name not in self.schema:
                raise ValueError(f"Unknown parameter: {name}")
            if not self.schema[name].validate(value):
                raise ValueError(f"Invalid value for parameter {name}: {value}")
            self._parameters[name] = value
        self._version += 1
    
    def get(self, name: str) -> Any:
        """Get a parameter value."""
        return self._parameters.get(name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return self._parameters.copy()
    
    @property
    def version(self) -> int:
        """Get the current version number."""
        return self._version


class StrategyBase(Component, ABC):
    """
    Abstract base class for all trading strategies.
    
    Provides core functionality for event handling, parameter management,
    and signal generation while enforcing a consistent interface.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.parameters = self._create_parameter_set()
        self._signals: List[Signal] = []
        self._subscriptions: Set[str] = set()
        
    @abstractmethod
    def _create_parameter_set(self) -> ParameterSet:
        """Create and return the parameter set for this strategy."""
        pass
    
    @abstractmethod
    def calculate_signals(self, event: Event) -> List[Signal]:
        """
        Calculate trading signals based on the given event.
        
        Args:
            event: The event to process (typically a BAR event)
            
        Returns:
            List of signals generated
        """
        pass
    
    def on_event(self, event: Event) -> None:
        """Handle incoming events."""
        if event.event_type == "BAR":
            signals = self.calculate_signals(event)
            for signal in signals:
                self._signals.append(signal)
                self._emit_signal(signal)
    
    def _emit_signal(self, signal: Signal) -> None:
        """Emit a signal event."""
        signal_event = Event(
            event_type="SIGNAL",
            data={
                "timestamp": signal.timestamp,
                "symbol": signal.symbol,
                "signal_type": signal.signal_type.value,
                "strength": signal.strength,
                "strategy": self.name,
                "rule_name": signal.rule_name,
                "metadata": signal.metadata
            }
        )
        self.emit_event(signal_event)
    
    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self._signals.clear()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current strategy state."""
        state = super().get_state()
        state.update({
            "parameters": self.parameters.to_dict(),
            "parameter_version": self.parameters.version,
            "signal_count": len(self._signals)
        })
        return state


class CompositeStrategy(StrategyBase):
    """
    Combines multiple strategies with weighted signal aggregation.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.strategies: Dict[str, StrategyBase] = {}
        self.weights: Dict[str, float] = {}
        self.aggregation_method = config.get("aggregation_method", "weighted_average")
    
    def add_strategy(self, strategy: StrategyBase, weight: float = 1.0) -> None:
        """Add a strategy with a given weight."""
        self.strategies[strategy.name] = strategy
        self.weights[strategy.name] = weight
    
    def _create_parameter_set(self) -> ParameterSet:
        """Create composite parameter set."""
        schema = {
            "min_agreement": ParameterSchema(
                name="min_agreement",
                type=float,
                default=0.6,
                min_value=0.0,
                max_value=1.0,
                description="Minimum agreement ratio for signal generation"
            )
        }
        return ParameterSet(schema)
    
    def calculate_signals(self, event: Event) -> List[Signal]:
        """Aggregate signals from all strategies."""
        all_signals = []
        
        for strategy in self.strategies.values():
            signals = strategy.calculate_signals(event)
            all_signals.extend(signals)
        
        # Aggregate based on method
        if self.aggregation_method == "weighted_average":
            return self._weighted_average_aggregation(all_signals, event.data["timestamp"])
        elif self.aggregation_method == "majority_vote":
            return self._majority_vote_aggregation(all_signals, event.data["timestamp"])
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def _weighted_average_aggregation(self, signals: List[Signal], timestamp: datetime) -> List[Signal]:
        """Aggregate signals using weighted average."""
        if not signals:
            return []
        
        # Group by symbol
        symbol_signals: Dict[str, List[Tuple[Signal, float]]] = {}
        for signal in signals:
            strategy_name = signal.metadata.get("strategy", "unknown")
            weight = self.weights.get(strategy_name, 1.0)
            
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = []
            symbol_signals[signal.symbol].append((signal, weight))
        
        # Calculate weighted signals
        aggregated_signals = []
        for symbol, weighted_signals in symbol_signals.items():
            total_weight = sum(w for _, w in weighted_signals)
            if total_weight == 0:
                continue
                
            weighted_strength = sum(s.strength * w for s, w in weighted_signals) / total_weight
            
            # Determine signal type based on strength
            if weighted_strength > 0.1:
                signal_type = SignalType.BUY
            elif weighted_strength < -0.1:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            aggregated_signals.append(Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=signal_type,
                strength=weighted_strength,
                rule_name="composite",
                metadata={"aggregation_method": "weighted_average"}
            ))
        
        return aggregated_signals
    
    def _majority_vote_aggregation(self, signals: List[Signal], timestamp: datetime) -> List[Signal]:
        """Aggregate signals using majority vote."""
        # Implementation would go here
        raise NotImplementedError("Majority vote aggregation not yet implemented")


class RuleBase(ABC):
    """Base class for individual trading rules."""
    
    def __init__(self, name: str, parameters: Dict[str, Any]):
        self.name = name
        self.parameters = parameters
        self.enabled = True
    
    @abstractmethod
    def evaluate(self, data: Dict[str, Any]) -> Optional[Signal]:
        """Evaluate the rule and return a signal if conditions are met."""
        pass
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable this rule."""
        self.enabled = enabled


class IndicatorBase(ABC):
    """Base class for technical indicators."""
    
    def __init__(self, period: int):
        self.period = period
        self._values: List[float] = []
        self._timestamps: List[datetime] = []
    
    @abstractmethod
    def calculate(self, value: float, timestamp: datetime) -> Optional[float]:
        """Calculate the indicator value."""
        pass
    
    def get_current_value(self) -> Optional[float]:
        """Get the most recent indicator value."""
        return self._values[-1] if self._values else None
    
    def reset(self) -> None:
        """Reset the indicator state."""
        self._values.clear()
        self._timestamps.clear()