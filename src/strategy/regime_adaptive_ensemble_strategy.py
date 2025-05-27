# src/strategy/regime_adaptive_ensemble_strategy.py
"""
Regime Adaptive Ensemble Strategy

Combines MA and RSI signals with regime-adaptive parameter switching.
This gives us the best of both worlds:
1. Multiple signal types (ensemble)
2. Dynamic parameter adaptation based on market regime
"""

import logging
from typing import Dict, Any, Optional, List
from collections import deque

from src.strategy.regime_adaptive_strategy import RegimeAdaptiveStrategy
from src.strategy.components.indicators.oscillators import RSIIndicator
from src.strategy.components.rules.rsi_rules import RSIRule
from src.core.event import Event, EventType
from src.core.exceptions import ConfigurationError


class RegimeAdaptiveEnsembleStrategy(RegimeAdaptiveStrategy):
    """
    An ensemble strategy that combines MA and RSI signals with regime-adaptive parameters.
    
    This strategy:
    - Uses both MA crossover and RSI signals
    - Weights the signals to make trading decisions
    - Automatically adapts parameters based on detected market regime
    - Loads optimal parameters for each regime from configuration file
    """
    
    def _initialize(self):
        """Initialize the ensemble strategy with RSI components."""
        # Call parent's _initialize to set up MA and regime adaptation
        super()._initialize()
        
        # Initialize RSI components
        self._initialize_rsi_components()
        
        # Load weights from config
        self._ma_weight = self.get_specific_config('ma_weight', 0.5)
        self._rsi_weight = self.get_specific_config('rsi_weight', 0.5)
        
        # Normalize weights
        total = self._ma_weight + self._rsi_weight
        if total > 0:
            self._ma_weight /= total
            self._rsi_weight /= total
            
        self.logger.info(
            f"{self.instance_name} initialized as ensemble strategy with "
            f"MA weight: {self._ma_weight:.2f}, RSI weight: {self._rsi_weight:.2f}"
        )
        
    def _initialize_rsi_components(self):
        """Initialize RSI indicator and rule."""
        # Get RSI configuration
        rsi_period = self.get_specific_config('rsi_indicator.period', 14)
        oversold = self.get_specific_config('rsi_rule.oversold_threshold', 30.0)
        overbought = self.get_specific_config('rsi_rule.overbought_threshold', 70.0)
        
        # Create RSI indicator
        self.rsi_indicator = RSIIndicator(
            instance_name=f"{self.instance_name}_rsi_indicator",
            config_key=None
        )
        self.rsi_indicator.initialize(self._context)
        self.rsi_indicator._period = rsi_period
        self.rsi_indicator._prices = deque(maxlen=rsi_period + 1)
        
        # Create RSI rule
        self.rsi_rule = RSIRule(
            instance_name=f"{self.instance_name}_rsi_rule",
            config_key=None
        )
        self.rsi_rule.initialize(self._context)
        self.rsi_rule._oversold_threshold = oversold
        self.rsi_rule._overbought_threshold = overbought
        self.rsi_rule._rsi_indicator = self.rsi_indicator
        
        self.logger.info(
            f"RSI components initialized: period={rsi_period}, "
            f"oversold={oversold}, overbought={overbought}"
        )
        
    def set_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Set parameters including RSI parameters and weights.
        """
        # First call parent to handle MA parameters
        success = super().set_parameters(params)
        
        # Handle RSI indicator parameters
        if "rsi_indicator.period" in params:
            new_period = int(params["rsi_indicator.period"])
            if hasattr(self, 'rsi_indicator'):
                self.rsi_indicator._period = new_period
                self.rsi_indicator._prices = deque(maxlen=new_period + 1)
                self.logger.info(f"Updated RSI period to: {new_period}")
        
        # Handle RSI rule parameters
        if "rsi_rule.oversold_threshold" in params:
            if hasattr(self, 'rsi_rule'):
                self.rsi_rule._oversold_threshold = float(params["rsi_rule.oversold_threshold"])
                self.logger.info(f"Updated RSI oversold to: {self.rsi_rule._oversold_threshold}")
                
        if "rsi_rule.overbought_threshold" in params:
            if hasattr(self, 'rsi_rule'):
                self.rsi_rule._overbought_threshold = float(params["rsi_rule.overbought_threshold"])
                self.logger.info(f"Updated RSI overbought to: {self.rsi_rule._overbought_threshold}")
        
        # Handle weights
        if "ma_weight" in params or "ma_rule.weight" in params:
            self._ma_weight = float(params.get("ma_weight", params.get("ma_rule.weight", self._ma_weight)))
            
        if "rsi_weight" in params or "rsi_rule.weight" in params:
            self._rsi_weight = float(params.get("rsi_weight", params.get("rsi_rule.weight", self._rsi_weight)))
            
        # Normalize weights
        total = self._ma_weight + self._rsi_weight
        if total > 0:
            self._ma_weight /= total
            self._rsi_weight /= total
            
        self.logger.info(f"Updated weights: MA={self._ma_weight:.2f}, RSI={self._rsi_weight:.2f}")
        
        return success
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get all parameters including RSI."""
        params = super().get_parameters()
        
        # Add RSI parameters
        if hasattr(self, 'rsi_indicator'):
            params["rsi_indicator.period"] = self.rsi_indicator._period
            
        if hasattr(self, 'rsi_rule'):
            params["rsi_rule.oversold_threshold"] = self.rsi_rule._oversold_threshold
            params["rsi_rule.overbought_threshold"] = self.rsi_rule._overbought_threshold
            
        # Add weights
        params["ma_rule.weight"] = self._ma_weight
        params["rsi_rule.weight"] = self._rsi_weight
        
        return params
        
    def get_optimizable_parameters(self) -> Dict[str, Any]:
        """Get optimizable parameters for the ensemble strategy."""
        # Get MA parameters from parent
        params = super().get_optimizable_parameters()
        
        # Add RSI and weight parameters
        if hasattr(self, 'rsi_indicator'):
            params["rsi_indicator.period"] = self.rsi_indicator._period
            
        if hasattr(self, 'rsi_rule'):
            params["rsi_rule.oversold_threshold"] = self.rsi_rule._oversold_threshold
            params["rsi_rule.overbought_threshold"] = self.rsi_rule._overbought_threshold
            
        params["ma_rule.weight"] = self._ma_weight
        params["rsi_rule.weight"] = self._rsi_weight
        
        return params
        
    def _on_bar_event(self, event: Event):
        """
        Handle bar events by generating ensemble signals.
        """
        if event.event_type != EventType.BAR or event.payload.get("symbol") != self._symbol:
            return
            
        bar_data = event.payload
        close_price = float(bar_data.get("close"))
        timestamp = bar_data.get("timestamp")
        
        if close_price is None or timestamp is None:
            return
            
        # Update price history for MA
        self._prices.append(close_price)
        
        # Calculate MAs
        current_short_ma, current_long_ma = self._calculate_mas()
        
        # Get MA signal
        ma_signal = self._get_ma_signal(current_short_ma, current_long_ma)
        
        # Update RSI
        if hasattr(self, 'rsi_indicator'):
            self.rsi_indicator.update(close_price, timestamp)
            
        # Get RSI signal
        rsi_signal = self._get_rsi_signal(bar_data)
        
        # Combine signals
        combined_signal = self._combine_signals(ma_signal, rsi_signal)
        
        # Update MA state for next bar
        if current_short_ma is not None:
            self._prev_short_ma = current_short_ma
        if current_long_ma is not None:
            self._prev_long_ma = current_long_ma
            
        # Emit signal if needed
        if combined_signal != 0 and combined_signal != self._current_signal_state:
            self._emit_signal(combined_signal, close_price, timestamp)
            self._current_signal_state = combined_signal
            
    def _calculate_mas(self):
        """Calculate current MA values."""
        current_short_ma = None
        current_long_ma = None
        
        if len(self._prices) >= self._short_window:
            current_short_ma = sum(list(self._prices)[-self._short_window:]) / self._short_window
            
        if len(self._prices) >= self._long_window:
            current_long_ma = sum(self._prices) / len(self._prices)
            
        return current_short_ma, current_long_ma
        
    def _get_ma_signal(self, current_short_ma, current_long_ma):
        """Get MA crossover signal."""
        if (current_short_ma is None or current_long_ma is None or
            self._prev_short_ma is None or self._prev_long_ma is None):
            return 0
            
        # Bullish crossover
        if self._prev_short_ma <= self._prev_long_ma and current_short_ma > current_long_ma:
            return 1
            
        # Bearish crossover
        if self._prev_short_ma >= self._prev_long_ma and current_short_ma < current_long_ma:
            return -1
            
        return 0
        
    def _get_rsi_signal(self, bar_data):
        """Get RSI signal."""
        if not hasattr(self, 'rsi_rule') or not self.rsi_indicator._current_value:
            return 0
            
        triggered, strength, signal_type = self.rsi_rule.evaluate(bar_data)
        
        if triggered:
            return int(strength)  # 1 for BUY, -1 for SELL
            
        return 0
        
    def _combine_signals(self, ma_signal, rsi_signal):
        """Combine MA and RSI signals using weights."""
        # Calculate weighted signal strength
        ma_weighted = ma_signal * self._ma_weight
        rsi_weighted = rsi_signal * self._rsi_weight
        combined_strength = ma_weighted + rsi_weighted
        
        # Generate signal based on strength threshold
        if abs(combined_strength) >= 0.3:  # Threshold for signal generation
            return 1 if combined_strength > 0 else -1
            
        return 0
        
    def _emit_signal(self, signal_type, price, timestamp):
        """Emit trading signal."""
        signal_payload = {
            "symbol": self._symbol,
            "timestamp": timestamp,
            "signal_type": signal_type,
            "price_at_signal": price,
            "strategy_id": self.instance_name,
            "reason": f"Ensemble(MA:{self._ma_weight:.2f},RSI:{self._rsi_weight:.2f},Regime:{self._current_regime})",
            "regime": self._current_regime
        }
        
        signal_event = Event(EventType.SIGNAL, signal_payload)
        self.event_bus.publish(signal_event)
        
        self.logger.info(
            f"Signal: {'BUY' if signal_type > 0 else 'SELL'} at {price:.2f} "
            f"(Regime: {self._current_regime}, Weights: MA={self._ma_weight:.2f}, RSI={self._rsi_weight:.2f})"
        )