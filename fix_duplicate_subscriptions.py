#!/usr/bin/env python3
# fix_duplicate_subscriptions.py - Fix duplicate event subscriptions in ADMF

import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to basic_portfolio.py
portfolio_file = "/Users/daws/ADMF/src/risk/basic_portfolio.py"
strategy_file = "/Users/daws/ADMF/src/strategy/regime_adaptive_strategy.py"

# Modification for BasicPortfolio
def fix_portfolio_class():
    logger.info(f"Modifying BasicPortfolio in {portfolio_file}")
    
    # Read the file
    with open(portfolio_file, 'r') as f:
        content = f.read()
    
    # Modify the reset method to unsubscribe from events
    reset_method = """    def reset(self):
        """Reset the portfolio to its initial state for a fresh backtest run."""
        self.logger.info(f"Resetting portfolio '{self.name}' to initial state")
        
        # Unsubscribe from events first to prevent duplicate subscriptions
        if self._event_bus:
            try:
                self._event_bus.unsubscribe(EventType.FILL, self.on_fill)
                self._event_bus.unsubscribe(EventType.BAR, self.on_bar)
                self._event_bus.unsubscribe(EventType.CLASSIFICATION, self.on_classification_change)
                self.logger.debug(f"'{self.name}' unsubscribed from events during reset.")
            except Exception as e:
                self.logger.warning(f"Error unsubscribing from events during reset: {e}")
        
        # Close any open positions
        if self.open_positions:
            now = datetime.datetime.now(datetime.timezone.utc)
            self.close_all_positions(now)
            
        # Reset cash and positions
        self.current_cash = self.initial_cash
        self.open_positions = {}
        self._trade_log = []
        
        # Reset performance metrics
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.current_holdings_value = 0.0
        self.current_total_value = self.initial_cash
        
        # Reset market data
        self._last_bar_prices = {}
        
        # Reset history
        self._portfolio_value_history = []
        
        # Reset market regime to default
        self._current_market_regime = "default"
        
        self.logger.info(f"Portfolio '{self.name}' reset successfully. Cash: {self.current_cash:.2f}, Total Value: {self.current_total_value:.2f}")"""
    
    # Find and replace the reset method
    original_reset = """    def reset(self):
        """Reset the portfolio to its initial state for a fresh backtest run."""
        self.logger.info(f"Resetting portfolio '{self.name}' to initial state")
        
        # Close any open positions
        if self.open_positions:
            now = datetime.datetime.now(datetime.timezone.utc)
            self.close_all_positions(now)
            
        # Reset cash and positions
        self.current_cash = self.initial_cash
        self.open_positions = {}
        self._trade_log = []
        
        # Reset performance metrics
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.current_holdings_value = 0.0
        self.current_total_value = self.initial_cash
        
        # Reset market data
        self._last_bar_prices = {}
        
        # Reset history
        self._portfolio_value_history = []
        
        # Reset market regime to default
        self._current_market_regime = "default"
        
        self.logger.info(f"Portfolio '{self.name}' reset successfully. Cash: {self.current_cash:.2f}, Total Value: {self.current_total_value:.2f}")"""
    
    # Check if the content already has the fix
    if "# Unsubscribe from events first to prevent duplicate subscriptions" in content:
        logger.info("BasicPortfolio already has the fix applied.")
        return False
    
    # Apply the changes
    new_content = content.replace(original_reset, reset_method)
    
    # Make sure the replacement worked
    if new_content == content:
        logger.error("Could not find reset method to replace in BasicPortfolio.")
        return False
    
    # Write the changes
    with open(portfolio_file, 'w') as f:
        f.write(new_content)
    
    logger.info("Successfully modified BasicPortfolio reset method.")
    return True

# Modification for RegimeAdaptiveStrategy
def fix_strategy_class():
    logger.info(f"Modifying RegimeAdaptiveStrategy in {strategy_file}")
    
    # Read the file
    with open(strategy_file, 'r') as f:
        content = f.read()
    
    # Add a flag to track subscription status
    if "_is_subscribed_to_classification" not in content:
        # Find the __init__ method
        init_end = content.find("self.logger.info(", content.find("def __init__"))
        if init_end == -1:
            logger.error("Could not find __init__ method in RegimeAdaptiveStrategy.")
            return False
        
        # Find the end of the __init__ method to add our flag
        next_line_end = content.find("\n", init_end)
        if next_line_end == -1:
            logger.error("Could not find end of __init__ method in RegimeAdaptiveStrategy.")
            return False
        
        # Add our flag
        new_content = content[:next_line_end+1] + "        self._is_subscribed_to_classification = False\n" + content[next_line_end+1:]
        
        # Update start method to check subscription status
        start_method = """    def start(self):
        """
        Overridden to subscribe to classification events.
        """
        super().start()
        
        # Subscribe to classification events to adapt to regime changes
        if self._event_bus and not self._is_subscribed_to_classification:
            from src.core.event import EventType
            self._event_bus.subscribe(EventType.CLASSIFICATION, self.on_classification_change)
            self._is_subscribed_to_classification = True
            self.logger.info(f"'{self.name}' subscribed to CLASSIFICATION events.")"""
        
        original_start = """    def start(self):
        """
        Overridden to subscribe to classification events.
        """
        super().start()
        
        # Subscribe to classification events to adapt to regime changes
        if self._event_bus:
            from src.core.event import EventType
            self._event_bus.subscribe(EventType.CLASSIFICATION, self.on_classification_change)
            self.logger.info(f"'{self.name}' subscribed to CLASSIFICATION events.")"""
        
        # Replace the start method
        new_content = new_content.replace(original_start, start_method)
        
        # Update stop method to reset subscription status
        stop_method = """    def stop(self):
        """
        Overridden to unsubscribe from classification events.
        """
        # Unsubscribe from classification events
        if self._event_bus and self._is_subscribed_to_classification:
            try:
                from src.core.event import EventType
                self._event_bus.unsubscribe(EventType.CLASSIFICATION, self.on_classification_change)
                self._is_subscribed_to_classification = False
                self.logger.info(f"'{self.name}' unsubscribed from CLASSIFICATION events.")
            except Exception as e:
                self.logger.error(f"Error unsubscribing '{self.name}' from CLASSIFICATION events: {e}", exc_info=True)
        super().stop()"""
        
        # Find the stop method
        stop_start = new_content.find("def stop(self):")
        if stop_start == -1:
            logger.error("Could not find stop method in RegimeAdaptiveStrategy.")
            return False
        
        # Find the end of the stop method declaration
        stop_end = new_content.find("\n", stop_start)
        if stop_end == -1:
            logger.error("Could not find end of stop method declaration in RegimeAdaptiveStrategy.")
            return False
        
        # Find the docstring end
        docstring_end = new_content.find('"""', new_content.find('"""', stop_end) + 3)
        if docstring_end == -1:
            logger.error("Could not find docstring end in stop method of RegimeAdaptiveStrategy.")
            return False
        
        # Find the end of the docstring line
        docstring_line_end = new_content.find('\n', docstring_end)
        if docstring_line_end == -1:
            logger.error("Could not find end of docstring line in stop method of RegimeAdaptiveStrategy.")
            return False
        
        # Replace the content from the docstring end to the next super().stop() call
        super_stop = new_content.find("super().stop()", docstring_line_end)
        if super_stop == -1:
            logger.error("Could not find super().stop() call in RegimeAdaptiveStrategy.")
            return False
        
        # Extract the current stop method implementation without the method declaration or docstring
        current_implementation = new_content[docstring_line_end+1:super_stop].strip()
        
        # Create the new stop method by replacing the current implementation with our new one
        new_stop_method = new_content[:docstring_line_end+1] + "\n        # Unsubscribe from classification events\n        if self._event_bus and self._is_subscribed_to_classification:\n            try:\n                from src.core.event import EventType\n                self._event_bus.unsubscribe(EventType.CLASSIFICATION, self.on_classification_change)\n                self._is_subscribed_to_classification = False\n                self.logger.info(f\"'{self.name}' unsubscribed from CLASSIFICATION events.\")\n            except Exception as e:\n                self.logger.error(f\"Error unsubscribing '{self.name}' from CLASSIFICATION events: {e}\", exc_info=True)\n        " + new_content[super_stop:]
        
        # Use the new version
        new_content = new_stop_method
        
        # Write the changes
        with open(strategy_file, 'w') as f:
            f.write(new_content)
        
        logger.info("Successfully modified RegimeAdaptiveStrategy subscription management.")
        return True
    else:
        logger.info("RegimeAdaptiveStrategy already has subscription tracking flag.")
        return False

if __name__ == "__main__":
    logger.info("Starting to fix duplicate subscription issues")
    
    fixed_portfolio = fix_portfolio_class()
    fixed_strategy = fix_strategy_class()
    
    if fixed_portfolio or fixed_strategy:
        logger.info("Fixes applied successfully. Please run your application again and check for warnings.")
    else:
        logger.info("No fixes were needed or applied.")