"""
Improved crossover rule that only signals on actual crossovers.
"""

from typing import Dict, Any, Optional
from ..rule import RuleBase, RuleResult


class TrueCrossoverRule(RuleBase):
    """
    Moving average crossover rule that only signals on actual crossover events.
    
    Tracks previous state to detect when MAs actually cross.
    """
    
    def __init__(self, name: str = "True_MA_Crossover"):
        super().__init__(name)
        self._parameters = {
            'generate_exit_signals': True,
            'min_separation': 0.0001  # Minimum separation to trigger signal
        }
        self._prev_fast_value: Optional[float] = None
        self._prev_slow_value: Optional[float] = None
        self._prev_position: Optional[str] = None  # 'above', 'below', or None
        
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
            
        # Determine current position
        current_position = 'above' if fast_value > slow_value else 'below'
        
        # Calculate separation
        separation = abs(fast_value - slow_value) / slow_value if slow_value != 0 else 0
        min_sep = self._parameters.get('min_separation', 0.0001)
        
        # Initialize previous values if first time
        if self._prev_position is None:
            self._prev_position = current_position
            self._prev_fast_value = fast_value
            self._prev_slow_value = slow_value
            return RuleResult(signal=0, strength=0.0, reason="Initializing")
            
        # Check for crossover
        signal = 0
        strength = 0.0
        reason = "No crossover"
        
        if current_position != self._prev_position and separation > min_sep:
            if current_position == 'above':
                # Bullish crossover (fast crossed above slow)
                signal = 1
                strength = min(1.0, separation / 0.01)  # Max strength at 1% separation
                reason = f"Bullish crossover (fast {fast_value:.2f} > slow {slow_value:.2f})"
            else:
                # Bearish crossover (fast crossed below slow)
                if self._parameters.get('generate_exit_signals', True):
                    signal = -1
                    strength = min(1.0, separation / 0.01)
                    reason = f"Bearish crossover (fast {fast_value:.2f} < slow {slow_value:.2f})"
                    
        # Update state for next evaluation
        self._prev_position = current_position
        self._prev_fast_value = fast_value
        self._prev_slow_value = slow_value
        
        return RuleResult(signal=signal, strength=strength, reason=reason)
        
    def reset(self) -> None:
        """Reset rule state."""
        super().reset()
        self._prev_fast_value = None
        self._prev_slow_value = None
        self._prev_position = None