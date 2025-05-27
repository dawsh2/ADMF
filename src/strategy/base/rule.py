"""
Base class for trading rules.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

from ...core.component_base import ComponentBase
from .parameter import ParameterSpace, Parameter


@dataclass
class RuleResult:
    """Result from rule evaluation."""
    signal: int  # -1 (sell), 0 (neutral), 1 (buy)
    strength: float  # 0.0 to 1.0
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RuleBase(ComponentBase, ABC):
    """
    Base class for all trading rules.
    
    Rules evaluate market conditions and generate trading signals.
    They can depend on indicators and other data sources.
    Inherits from ComponentBase to provide standard lifecycle and optimization interface.
    """
    
    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        """Initialize rule with ComponentBase pattern."""
        super().__init__(instance_name, config_key)
        
        # Rule-specific state
        self._ready = False
        self._parameters: Dict[str, Any] = {}
        self._dependencies: Dict[str, Any] = {}  # Indicators or other data sources
        self._last_signal: Optional[int] = None
        self._signals_generated = 0
    
    def _initialize(self) -> None:
        """Component-specific initialization."""
        # Load any configuration
        self._load_config()
        # Reset state
        self.reset()
        
    def _load_config(self) -> None:
        """Load configuration from component_config."""
        # Subclasses can override to load specific config
        pass
        
    def _start(self) -> None:
        """Component-specific start logic."""
        self.logger.debug(f"Rule '{self.instance_name}' started")
        
    def _stop(self) -> None:
        """Component-specific stop logic."""
        self.reset()
        
    @property
    def name(self) -> str:
        """Unique name for this rule (compatibility property)."""
        return self.instance_name
        
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
        # Validate first
        valid, error = self.validate_parameters(params)
        if not valid:
            raise ValueError(f"Invalid parameters: {error}")
        
        self._parameters.update(params)
        # Reset on parameter change
        self.reset()
        
    def get_parameter_space(self) -> ParameterSpace:
        """Get parameter space for optimization."""
        # Base implementation - subclasses should override
        return ParameterSpace(f"{self.instance_name}_params")
        
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate parameters (can be overridden by subclasses)."""
        return True, None
        
    def apply_parameters(self, params: Dict[str, Any]) -> None:
        """Apply parameters to rule and its dependencies."""
        # Separate rule parameters from dependency parameters
        rule_params = {}
        dependency_params = {}
        
        for key, value in params.items():
            if '.' in key:
                # Namespaced parameter for dependency
                dep_name, param_name = key.split('.', 1)
                if dep_name not in dependency_params:
                    dependency_params[dep_name] = {}
                dependency_params[dep_name][param_name] = value
            else:
                # Direct rule parameter
                rule_params[key] = value
        
        # Apply rule parameters
        if rule_params:
            self.set_parameters(rule_params)
        
        # Apply dependency parameters
        for dep_name, dep_params in dependency_params.items():
            if dep_name in self._dependencies:
                dep = self._dependencies[dep_name]
                if hasattr(dep, 'apply_parameters'):
                    dep.apply_parameters(dep_params)
        
    def get_optimizable_parameters(self) -> Dict[str, Any]:
        """Get optimizable parameters including dependencies."""
        params = self.get_parameters()
        
        # Include dependency parameters with namespacing
        for dep_name, dep in self._dependencies.items():
            if hasattr(dep, 'get_optimizable_parameters'):
                dep_params = dep.get_optimizable_parameters()
                for param_name, param_value in dep_params.items():
                    params[f"{dep_name}.{param_name}"] = param_value
                    
        return params
        
    def reset(self) -> None:
        """Reset rule state."""
        self._ready = False
        self._last_signal = None
        self._signals_generated = 0
        

class CrossoverRule(RuleBase):
    """Moving average crossover rule."""
    
    def __init__(self, name: str = "MA_Crossover", config_key: Optional[str] = None):
        super().__init__(instance_name=name, config_key=config_key)
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
        
        # If we have dependent indicators, include their parameter spaces
        if 'fast_ma' in self._dependencies:
            fast_indicator = self._dependencies['fast_ma']
            if hasattr(fast_indicator, 'get_parameter_space'):
                fast_space = fast_indicator.get_parameter_space()
                # Add as subspace to maintain namespacing
                space.add_subspace('fast_ma', fast_space)
                
        if 'slow_ma' in self._dependencies:
            slow_indicator = self._dependencies['slow_ma']
            if hasattr(slow_indicator, 'get_parameter_space'):
                slow_space = slow_indicator.get_parameter_space()
                # Add as subspace to maintain namespacing
                space.add_subspace('slow_ma', slow_space)
        
        return space
        

class ThresholdRule(RuleBase):
    """Rule based on indicator threshold levels."""
    
    def __init__(self, name: str = "Threshold_Rule", config_key: Optional[str] = None):
        super().__init__(instance_name=name, config_key=config_key)
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
        
        # If we have a dependent indicator, include its parameter space
        indicator_name = self._parameters.get('indicator_name', 'indicator')
        if indicator_name in self._dependencies:
            indicator = self._dependencies[indicator_name]
            if hasattr(indicator, 'get_parameter_space'):
                indicator_space = indicator.get_parameter_space()
                # Add as subspace to maintain namespacing
                space.add_subspace(indicator_name, indicator_space)
        
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