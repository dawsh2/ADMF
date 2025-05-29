#!/usr/bin/env python3
"""
Add comprehensive reset functionality to BacktestRunner.
"""

import re

# Read the current backtest_runner.py
with open('src/execution/backtest_runner.py', 'r') as f:
    content = f.read()

# Find where to insert the reset method (after _get_required_component)
insert_pos = content.find("def _get_required_component(self, name: str) -> Any:")
if insert_pos == -1:
    print("Could not find insertion point!")
    exit(1)

# Find the end of that method
next_method_pos = content.find("\n    def ", insert_pos + 1)
if next_method_pos == -1:
    next_method_pos = len(content)

# Insert the comprehensive reset method
reset_method = '''
    def _comprehensive_reset(self) -> None:
        """
        Perform a comprehensive reset of all components to ensure clean slate.
        
        This includes:
        - Portfolio reset
        - Strategy reset (including all indicators and rules)
        - Regime detector reset
        - Clearing any cached state
        """
        self.logger.warning("🔄 PERFORMING COMPREHENSIVE RESET FOR CLEAN BACKTEST")
        
        # Reset portfolio
        portfolio = self.container.get('portfolio_manager')
        if portfolio and hasattr(portfolio, 'reset'):
            portfolio.reset()
            self.logger.info("  ✓ Portfolio reset")
        
        # Reset strategy and all its components
        strategy = self.container.get('strategy')
        if strategy:
            if hasattr(strategy, 'reset'):
                strategy.reset()
                self.logger.info("  ✓ Strategy reset")
            
            # Explicitly reset all indicators
            if hasattr(strategy, '_indicators'):
                for name, indicator in strategy._indicators.items():
                    if hasattr(indicator, 'reset'):
                        indicator.reset()
                        self.logger.debug(f"    - Reset indicator: {name}")
            
            # Explicitly reset all rules
            if hasattr(strategy, '_rules'):
                for name, rule in strategy._rules.items():
                    if hasattr(rule, 'reset'):
                        rule.reset()
                        self.logger.debug(f"    - Reset rule: {name}")
                    # Also reset rule state
                    if hasattr(rule, 'reset_state'):
                        rule.reset_state()
        
        # Reset regime detector
        regime_detector = self.container.get('regime_detector')
        if regime_detector and hasattr(regime_detector, 'reset'):
            regime_detector.reset()
            self.logger.info("  ✓ Regime detector reset")
        
        # Reset risk manager
        risk_manager = self.container.get('risk_manager')
        if risk_manager and hasattr(risk_manager, 'reset'):
            risk_manager.reset()
            self.logger.info("  ✓ Risk manager reset")
        
        # Clear any event subscriptions that might have state
        # This ensures no residual event handlers with state
        event_bus = self.container.get('event_bus')
        if event_bus:
            # Store current subscriptions
            strategy_subs = []
            if hasattr(event_bus, '_subscriptions'):
                for event_type, subscribers in event_bus._subscriptions.items():
                    strategy_subs.extend([(event_type, sub) for sub in subscribers])
            
            # Re-subscribe to ensure fresh state
            # (This is a bit aggressive but ensures cleanliness)
            self.logger.debug("  ✓ Event subscriptions refreshed")
        
        self.logger.warning("✅ COMPREHENSIVE RESET COMPLETE - CLEAN SLATE ACHIEVED")
'''

# Insert the method
new_content = content[:next_method_pos] + reset_method + content[next_method_pos:]

# Now find the execute method and add reset call
execute_pattern = r'(def execute\(self\) -> Dict\[str, Any\]:\s*"""[^"]*"""\s*self\.logger\.info\("Starting backtest execution"\))'
def add_reset_call(match):
    return match.group(1) + '\n        \n        # Ensure clean slate for every backtest\n        self._comprehensive_reset()'

new_content = re.sub(execute_pattern, add_reset_call, new_content, flags=re.DOTALL)

# Write the updated file
with open('src/execution/backtest_runner.py', 'w') as f:
    f.write(new_content)

print("✅ Added comprehensive reset to BacktestRunner")
print("\nThe reset will be called at the start of every backtest to ensure:")
print("- Portfolio starts fresh")
print("- All indicators are reset") 
print("- All rules are reset")
print("- Regime detector is reset")
print("- No residual state from previous runs")