"""
Base class for trading rules.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

from .strategy import StrategyComponent
from .parameter import ParameterSpace, Parameter


@dataclass
class RuleResult:
    """Result from rule evaluation."""
    signal: int  # -1 (sell), 0 (neutral), 1 (buy)
    strength: float  # 0.0 to 1.0
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RuleBase(StrategyComponent, ABC):
    """
    Base class for all trading rules.
    
    Rules evaluate market conditions and generate trading signals.
    They can depend on indicators and other data sources.
    """
    
    def __init__(self, name: str):
        self._name = name
        self._ready = False
        self._parameters: Dict[str, Any] = {}
        self._dependencies: Dict[str, Any] = {}  # Indicators or other data sources
        self._last_signal: Optional[int] = None
        self._signals_generated = 0
        
    @property
    def name(self) -> str:
        """Unique name for this rule."""
        return self._name
        
    @property
    def ready(self) -> bool:
        """Whether rule has enough data to produce valid signals."""
        return self._check_dependencies()
        
    def add_dependency(self, name: str, dependency: Any) -> None:
        """Add a dependency (usually an indicator) to this rule."""
        self._dependencies[name] = dependency
        
    def evaluate(self, bar_data: Dict[str, Any]) -> Tuple[int, float]:
        """
        Evaluate rule and return signal with strength.
        
        Returns:
            Tuple of (signal, strength) where:
            - signal: -1 (sell), 0 (neutral), 1 (buy)
            - strength: 0.0 to 1.0 confidence/strength
        """
        # Check if dependencies are ready
        if not self._check_dependencies():
            self._ready = False
            return 0, 0.0
            
        self._ready = True
        
        # Evaluate the rule
        result = self._evaluate_rule(bar_data)
        
        # Track signal
        if result.signal != 0:
            self._last_signal = result.signal
            self._signals_generated += 1
            
        return result.signal, result.strength
        
    @abstractmethod
    def _evaluate_rule(self, bar_data: Dict[str, Any]) -> RuleResult:
        """Implement rule logic. Must be overridden by subclasses."""
        pass
        
    def _check_dependencies(self) -> bool:
        """Check if all dependencies are ready."""
        for dep in self._dependencies.values():
            if hasattr(dep, 'ready') and not dep.ready:
                return False
        return True
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values."""
        return self._parameters.copy()
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set parameter values."""
        self._parameters.update(params)
        # Reset on parameter change
        self.reset()
        
    def get_parameter_space(self) -> ParameterSpace:
        """Get parameter space for optimization."""
        # Base implementation - subclasses should override
        return ParameterSpace(f"{self.name}_params")
        
    def reset(self) -> None:
        """Reset rule state."""
        self._ready = False
        self._last_signal = None
        self._signals_generated = 0
        

class CrossoverRule(RuleBase):
    """Moving average crossover rule."""
    
    def __init__(self, name: str = "MA_Crossover"):
        super().__init__(name)
        self._parameters = {
            'generate_exit_signals': True,
            'min_separation': 0.0001  # Minimum separation to trigger signal
        }
        
    def _evaluate_rule(self, bar_data: Dict[str, Any]) -> RuleResult:
        """Evaluate moving average crossover."""
        # Get indicators
        fast_ma = self._dependencies.get('fast_ma')
        slow_ma = self._dependencies.get('slow_ma')
        
        if not fast_ma or not slow_ma:
            return RuleResult(signal=0, strength=0.0, reason="Missing indicators")
            
        fast_value = fast_ma.value
        slow_value = slow_ma.value
        
        if fast_value is None or slow_value is None:
            return RuleResult(signal=0, strength=0.0, reason="Indicator values not ready")
            
        # Calculate separation
        separation = abs(fast_value - slow_value) / slow_value if slow_value != 0 else 0
        min_sep = self._parameters.get('min_separation', 0.0001)
        
        # Check for crossover
        if fast_value > slow_value and separation > min_sep:
            # Bullish crossover
            strength = min(1.0, separation / 0.01)  # Max strength at 1% separation
            return RuleResult(signal=1, strength=strength, reason="Bullish crossover")
            
        elif fast_value < slow_value and separation > min_sep:
            # Bearish crossover
            if self._parameters.get('generate_exit_signals', True):
                strength = min(1.0, separation / 0.01)
                return RuleResult(signal=-1, strength=strength, reason="Bearish crossover")
                
        return RuleResult(signal=0, strength=0.0, reason="No crossover")
        
    def get_parameter_space(self) -> ParameterSpace:
        """Get parameter space for optimization."""
        space = super().get_parameter_space()
        
        space.add_parameter(
            Parameter(
                name='generate_exit_signals',
                param_type='discrete',
                values=[True],  # Just test with exit signals enabled
                default=True
            )
        )
        
        space.add_parameter(
            Parameter(
                name='min_separation',
                param_type='discrete',
                values=[0.0001],  # Just test with one value
                default=0.0001
            )
        )
        
        return space
        

class ThresholdRule(RuleBase):
    """Rule based on indicator threshold levels."""
    
    def __init__(self, name: str = "Threshold_Rule"):
        super().__init__(name)
        self._parameters = {
            'buy_threshold': 30.0,
            'sell_threshold': 70.0,
            'indicator_name': 'indicator'
        }
        
    def _evaluate_rule(self, bar_data: Dict[str, Any]) -> RuleResult:
        """Evaluate threshold-based rule."""
        # Get indicator
        indicator_name = self._parameters.get('indicator_name', 'indicator')
        indicator = self._dependencies.get(indicator_name)
        
        if not indicator or not hasattr(indicator, 'value'):
            return RuleResult(signal=0, strength=0.0, reason="Missing indicator")
            
        value = indicator.value
        if value is None:
            return RuleResult(signal=0, strength=0.0, reason="Indicator not ready")
            
        buy_threshold = self._parameters.get('buy_threshold', 30.0)
        sell_threshold = self._parameters.get('sell_threshold', 70.0)
        
        # Check thresholds
        if value <= buy_threshold:
            # Buy signal - stronger the more extreme
            strength = (buy_threshold - value) / buy_threshold
            return RuleResult(signal=1, strength=strength, reason=f"Below buy threshold ({value:.2f})")
            
        elif value >= sell_threshold:
            # Sell signal - stronger the more extreme
            strength = (value - sell_threshold) / (100 - sell_threshold)
            return RuleResult(signal=-1, strength=strength, reason=f"Above sell threshold ({value:.2f})")
            
        return RuleResult(signal=0, strength=0.0, reason="Within neutral range")
        
    def get_parameter_space(self) -> ParameterSpace:
        """Get parameter space for optimization."""
        space = super().get_parameter_space()
        
        # For now, keep RSI thresholds fixed
        space.add_parameter(
            Parameter(
                name='buy_threshold',
                param_type='discrete',
                values=[30.0],  # Fixed value only
                default=30.0
            )
        )
        
        space.add_parameter(
            Parameter(
                name='sell_threshold',
                param_type='discrete',
                values=[70.0],  # Fixed value only
                default=70.0
            )
        )
        
        return space
        

class MomentumRule(RuleBase):
    """Rule based on price momentum."""
    
    def __init__(self, name: str = "Momentum_Rule"):
        super().__init__(name)
        self._parameters = {
            'lookback_period': 10,
            'momentum_threshold': 0.02  # 2% momentum threshold
        }
        self._price_history: List[float] = []
        
    def _evaluate_rule(self, bar_data: Dict[str, Any]) -> RuleResult:
        """Evaluate momentum-based rule."""
        current_price = float(bar_data.get('close', 0.0))
        
        # Update price history
        self._price_history.append(current_price)
        lookback = self._parameters.get('lookback_period', 10)
        
        if len(self._price_history) > lookback * 2:
            self._price_history = self._price_history[-lookback * 2:]
            
        # Need enough history
        if len(self._price_history) < lookback:
            return RuleResult(signal=0, strength=0.0, reason="Insufficient history")
            
        # Calculate momentum
        old_price = self._price_history[-lookback]
        if old_price == 0:
            return RuleResult(signal=0, strength=0.0, reason="Invalid historical price")
            
        momentum = (current_price - old_price) / old_price
        threshold = self._parameters.get('momentum_threshold', 0.02)
        
        # Generate signals based on momentum
        if momentum > threshold:
            strength = min(1.0, momentum / (threshold * 2))
            return RuleResult(
                signal=1, 
                strength=strength, 
                reason=f"Positive momentum: {momentum:.2%}"
            )
        elif momentum < -threshold:
            strength = min(1.0, abs(momentum) / (threshold * 2))
            return RuleResult(
                signal=-1, 
                strength=strength, 
                reason=f"Negative momentum: {momentum:.2%}"
            )
            
        return RuleResult(signal=0, strength=0.0, reason="Neutral momentum")
        
    def reset(self) -> None:
        """Reset rule state."""
        super().reset()
        self._price_history.clear()
        
    def get_parameter_space(self) -> ParameterSpace:
        """Get parameter space for optimization."""
        space = super().get_parameter_space()
        
        space.add_parameter(
            Parameter(
                name='lookback_period',
                param_type='discrete',
                values=[5, 10, 15, 20, 30],
                default=10
            )
        )
        
        space.add_parameter(
            Parameter(
                name='momentum_threshold',
                param_type='continuous',
                min_value=0.01,
                max_value=0.05,
                step=0.005,
                default=0.02
            )
        )
        
        return space