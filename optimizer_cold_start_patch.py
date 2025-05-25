#!/usr/bin/env python3
"""
Monkey patch for enhanced_optimizer to reset indicators before test.
Import this before running optimization to ensure cold start.
"""

from src.strategy.optimization.enhanced_optimizer import EnhancedOptimizer
import logging

# Store original method
_original_run_regime_adaptive_test = EnhancedOptimizer._run_regime_adaptive_test

def _reset_all_indicators(self):
    """Reset all indicators to cold state before test run."""
    self.logger.info("Resetting all indicators to cold state...")
    
    try:
        # Reset strategy indicators
        strategy = self._container.resolve(self._strategy_service_name)
        
        # Reset MA prices history
        if hasattr(strategy, '_prices'):
            strategy._prices = []
            self.logger.debug("Reset strategy price history")
        
        # Reset previous MA values
        if hasattr(strategy, '_prev_short_ma'):
            strategy._prev_short_ma = None
        if hasattr(strategy, '_prev_long_ma'):
            strategy._prev_long_ma = None
            
        # Reset bar counter
        if hasattr(strategy, '_bar_count'):
            strategy._bar_count = 0
            
        # Reset RSI indicator
        if hasattr(strategy, 'rsi_indicator'):
            if hasattr(strategy.rsi_indicator, 'reset'):
                strategy.rsi_indicator.reset()
            else:
                # Manual reset
                if hasattr(strategy.rsi_indicator, '_prices'):
                    strategy.rsi_indicator._prices = []
                if hasattr(strategy.rsi_indicator, '_current_value'):
                    strategy.rsi_indicator._current_value = None
            self.logger.debug("Reset RSI indicator")
        
        # Reset regime detector indicators
        regime_detector = self._container.resolve("MyPrimaryRegimeDetector")
        if hasattr(regime_detector, '_indicators'):
            for name, indicator in regime_detector._indicators.items():
                if hasattr(indicator, 'reset'):
                    indicator.reset()
                    self.logger.debug(f"Reset regime detector indicator: {name}")
                else:
                    # Manual reset for indicators without reset method
                    if hasattr(indicator, '_values'):
                        indicator._values = []
                    if hasattr(indicator, '_current_value'):
                        indicator._current_value = None
        
        # Reset regime detector state
        if hasattr(regime_detector, '_current_regime'):
            regime_detector._current_regime = 'default'
        if hasattr(regime_detector, '_regime_duration'):
            regime_detector._regime_duration = 0
            
        self.logger.info("✓ All indicators reset to cold state")
        
    except Exception as e:
        self.logger.error(f"Error resetting indicators: {e}", exc_info=True)

def _patched_run_regime_adaptive_test(self, optimized_params_by_regime, save_results=True):
    """Patched version that resets indicators before test."""
    
    # Get logger first
    logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
    
    # Call the reset method before running the original
    logger.info("PATCH: Resetting indicators before adaptive test")
    _reset_all_indicators(self)
    
    # Call original method
    return _original_run_regime_adaptive_test(self, optimized_params_by_regime, save_results)

# Apply the patch
EnhancedOptimizer._reset_all_indicators = _reset_all_indicators
EnhancedOptimizer._run_regime_adaptive_test = _patched_run_regime_adaptive_test

print("✓ Enhanced optimizer patched to reset indicators before test run")
