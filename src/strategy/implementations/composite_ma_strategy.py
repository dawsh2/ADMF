"""
Composite Moving Average Strategy using the new component architecture.
"""

from typing import Dict, Any, Optional

from ..base import (
    Strategy, 
    MovingAverageIndicator,
    RSIIndicator,
    CrossoverRule,
    ThresholdRule
)


class CompositeMAStrategy(Strategy):
    """
    Moving average strategy built using composition.
    
    Components:
    - Fast and slow moving averages
    - MA crossover rule
    - Optional RSI filter
    - Regime adaptation support
    """
    
    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        super().__init__(instance_name, config_key)
        
    def _initialize(self):
        """Initialize strategy components."""
        # Get configuration
        config = self.component_config or {}
        
        # Get parameters
        fast_period = config.get('fast_ma_period', 10)
        slow_period = config.get('slow_ma_period', 20)
        use_rsi_filter = config.get('use_rsi_filter', True)
        rsi_period = config.get('rsi_period', 14)
        
        # Create indicators
        fast_ma = MovingAverageIndicator(name="fast_ma", lookback_period=fast_period)
        slow_ma = MovingAverageIndicator(name="slow_ma", lookback_period=slow_period)
        
        # Add indicators
        self.add_indicator("fast_ma", fast_ma)
        self.add_indicator("slow_ma", slow_ma)
        
        # Create and add crossover rule
        crossover_rule = CrossoverRule(name="ma_crossover")
        crossover_rule.add_dependency("fast_ma", fast_ma)
        crossover_rule.add_dependency("slow_ma", slow_ma)
        self.add_rule("ma_crossover", crossover_rule, weight=1.0)
        
        # Optionally add RSI filter
        if use_rsi_filter:
            # Create RSI indicator
            rsi = RSIIndicator(name="rsi", lookback_period=rsi_period)
            self.add_indicator("rsi", rsi)
            
            # Create RSI threshold rule
            rsi_rule = ThresholdRule(name="rsi_filter")
            rsi_rule.add_dependency("indicator", rsi)
            rsi_rule.set_parameters({
                'buy_threshold': config.get('rsi_buy_threshold', 30),
                'sell_threshold': config.get('rsi_sell_threshold', 70),
                'indicator_name': 'indicator'
            })
            
            # Add with lower weight as a filter
            self.add_rule("rsi_filter", rsi_rule, weight=0.3)
            
        # Set aggregation method
        self._aggregation_method = config.get('aggregation_method', 'weighted')
        
        # Call parent initialization
        super()._initialize()
        
        self.logger.info(f"CompositeMAStrategy '{self.instance_name}' initialized: "
                        f"fast={fast_period}, slow={slow_period}, "
                        f"rsi_filter={use_rsi_filter}")
                        
    def _on_regime_change(self, old_regime: str, new_regime: str) -> None:
        """Handle regime changes by adjusting parameters."""
        # Check if we have regime-specific parameters
        if hasattr(self, '_regime_parameters') and new_regime in self._regime_parameters:
            regime_params = self._regime_parameters[new_regime]
            self.set_parameters(regime_params)
            self.logger.info(f"Applied {new_regime} regime parameters")
            
            
class AdaptiveCompositeStrategy(CompositeMAStrategy):
    """
    Extended version with regime-specific parameter adaptation.
    """
    
    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        super().__init__(instance_name, config_key)
        self._regime_parameters: Dict[str, Dict[str, Any]] = {}
        
    def _initialize(self):
        """Initialize with regime adaptation."""
        super()._initialize()
        
        # Load regime-specific parameters if available
        config = self.component_config or {}
        regime_params = config.get('regime_parameters', {})
        
        for regime, params in regime_params.items():
            self._regime_parameters[regime] = params
            
        self.logger.info(f"Loaded parameters for {len(self._regime_parameters)} regimes")
        
    def set_regime_parameters(self, regime: str, parameters: Dict[str, Any]) -> None:
        """Set parameters for a specific regime."""
        self._regime_parameters[regime] = parameters
        
        # If this is the current regime, apply immediately
        if regime == self._current_regime:
            self.set_parameters(parameters)
            
    def get_regime_parameters(self, regime: str) -> Dict[str, Any]:
        """Get parameters for a specific regime."""
        return self._regime_parameters.get(regime, {})
        

def create_ma_strategy(config: Dict[str, Any]) -> Strategy:
    """Factory function to create MA strategy from config."""
    strategy_type = config.get('type', 'composite')
    instance_name = config.get('name', 'ma_strategy')
    
    if strategy_type == 'adaptive':
        strategy = AdaptiveCompositeStrategy(instance_name)
    else:
        strategy = CompositeMAStrategy(instance_name)
        
    # Initialize with config
    strategy.component_config = config
    strategy._initialize()
    
    return strategy