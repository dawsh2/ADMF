#!/usr/bin/env python3
"""
Add debug logging to track RegimeDetector behavior.
"""

print("""
To debug the RegimeDetector mismatch, add this logging to RegimeDetector:

1. In RegimeDetector.on_bar(), add at the start:
   ```python
   if self._bars_processed < 50:  # Log first 50 bars
       self.logger.info(f"BAR {self._bars_processed}: {event.symbol} @ {event.timestamp}")
       self.logger.info(f"  Indicators - RSI: {self.indicators['rsi_14'].value}, "
                       f"ATR: {self.indicators['atr_20'].value}, "
                       f"Trend: {self.indicators['trend_10_30'].value}")
       self.logger.info(f"  Current regime: {self.current_classification}")
   ```

2. In RegimeDetector._check_regime_change(), add:
   ```python
   if new_regime != self.current_classification:
       self.logger.warning(f"REGIME CHANGE at bar {self._bars_processed}: "
                          f"{self.current_classification} -> {new_regime}")
   ```

3. In BacktestEngine._setup_components(), add:
   ```python
   # Log component states before setup
   regime_detector = components.get('regime_detector')
   if regime_detector:
       self.logger.info(f"RegimeDetector state before setup: {regime_detector.get_state()}")
       if hasattr(regime_detector, 'indicators'):
           for name, ind in regime_detector.indicators.items():
               self.logger.info(f"  {name}: warmup={ind.min_values}, current_count={ind.bar_count}")
   ```

This will help us see:
- Initial indicator values
- When regimes change
- Any initialization differences

The key issue is likely that the production run starts with "cold" indicators while
the OOS test might have some warmup from the training phase.
""")