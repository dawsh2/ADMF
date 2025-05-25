#!/usr/bin/env python3
"""
Extended ensemble strategy that handles warmup phase to match optimizer behavior.

This wrapper adds warmup capability to the existing ensemble strategy.
"""

from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy
from src.core.event import Event, EventType
import logging

class WarmupEnsembleStrategy(EnsembleSignalStrategy):
    """
    Ensemble strategy with warmup phase handling.
    
    During warmup:
    - Processes bars to update indicators
    - Suppresses signal generation
    - Tracks when to switch to evaluation mode
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._warmup_bars = 798  # Number of training bars
        self._bars_processed = 0
        self._in_warmup = True
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
    def handle_event(self, event: Event):
        """Override to handle warmup phase."""
        if event.event_type == EventType.BAR and event.data.get('symbol') == self._symbol:
            self._bars_processed += 1
            
            # Check if we're still in warmup
            if self._bars_processed <= self._warmup_bars:
                if self._bars_processed == 1:
                    self.logger.info(f"Starting warmup phase for {self._warmup_bars} bars...")
                elif self._bars_processed == self._warmup_bars:
                    self.logger.info(f"Warmup complete! Processed {self._warmup_bars} bars. Starting evaluation...")
                    self._in_warmup = False
                
                # Process the bar to update indicators but don't generate signals
                bar_data = event.data
                price = bar_data['close']
                
                # Update price history for MAs
                self._prices.append(price)
                
                # Update RSI indicator
                if hasattr(self, 'rsi_indicator'):
                    self.rsi_indicator.update(price)
                
                # Skip signal generation during warmup
                return
            
        # Normal processing after warmup
        super().handle_event(event)

# Create a monkey patch function to use this in production
def patch_ensemble_strategy():
    """
    Monkey patch the existing ensemble strategy to add warmup capability.
    """
    import src.strategy.implementations.ensemble_strategy as ensemble_module
    
    # Save original class
    original_class = ensemble_module.EnsembleSignalStrategy
    
    # Create patched version
    class PatchedEnsembleStrategy(original_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._warmup_bars = 798
            self._bars_processed = 0
            self._in_warmup = True
            
        def handle_event(self, event: Event):
            if event.event_type == EventType.BAR and event.data.get('symbol') == self._symbol:
                self._bars_processed += 1
                
                if self._bars_processed <= self._warmup_bars:
                    if self._bars_processed == 1:
                        self.logger.info(f"[WARMUP] Starting warmup phase for {self._warmup_bars} bars...")
                    elif self._bars_processed == self._warmup_bars:
                        self.logger.info(f"[WARMUP] Complete! Switching to evaluation mode...")
                        self._in_warmup = False
                    
                    # Update indicators during warmup
                    bar_data = event.data
                    price = bar_data['close']
                    self._prices.append(price)
                    
                    if hasattr(self, 'rsi_indicator'):
                        self.rsi_indicator.update(price)
                    
                    # Log progress every 100 bars
                    if self._bars_processed % 100 == 0:
                        self.logger.info(f"[WARMUP] Processed {self._bars_processed}/{self._warmup_bars} bars")
                    
                    return
            
            super().handle_event(event)
    
    # Replace the class in the module
    ensemble_module.EnsembleSignalStrategy = PatchedEnsembleStrategy
    
    print("Patched EnsembleSignalStrategy with warmup capability")

if __name__ == "__main__":
    print("Warmup-aware ensemble strategy ready.")
    print("\nTo use in production:")
    print("1. Import this module before creating strategies")
    print("2. Call patch_ensemble_strategy()")
    print("3. Run production with ALL data (no train/test split in data handler)")
    print("4. Strategy will warm up on first 798 bars, then evaluate on remaining 200")