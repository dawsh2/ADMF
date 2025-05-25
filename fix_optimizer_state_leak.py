#!/usr/bin/env python3
"""
Fix the state leak in optimizer by ensuring fresh components for adaptive test.
"""

print("FIXING OPTIMIZER STATE LEAK")
print("="*60)

print("\nTHE PROBLEM:")
print("- Grid search runs 24 backtests")
print("- Singleton components accumulate state")
print("- Adaptive test gets different results due to this state")
print("")

print("THE SOLUTION:")
print("Modify EnhancedOptimizerV2.run_adaptive_test to:")
print("1. Create a fresh container for the adaptive test")
print("2. Register new component instances")
print("3. Run the adaptive test with clean state")
print("")

print("IMPLEMENTATION:")
print("-"*60)

code_fix = '''
# In enhanced_optimizer_v2.py, modify run_adaptive_test:

def run_adaptive_test(self, results_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Run the adaptive test with FRESH components."""
    
    # Create a NEW container for clean state
    fresh_container = Container()
    fresh_event_bus = EventBus()
    
    # Register fresh components (same as main.py)
    # ... registration code ...
    
    # Create NEW BacktestEngine with fresh components
    fresh_engine = BacktestEngine(fresh_container, self.config_loader, fresh_event_bus)
    
    # Run adaptive test with clean state
    metric_value, regime_performance = fresh_engine.run_backtest(...)
'''

print(code_fix)

print("\nALTERNATIVE SOLUTIONS:")
print("-"*60)

print("1. Use Transient Components:")
print("   Change container.register_type(..., True, ...) to")
print("   container.register_type(..., False, ...) for all components")
print("")

print("2. Deep Reset Between Runs:")
print("   Implement deeper reset methods that clear ALL state")
print("   including static variables, caches, etc.")
print("")

print("3. Process Isolation:")
print("   Run each backtest in a separate process")
print("   (more complex but guarantees isolation)")
print("")

print("RECOMMENDED:")
print("-"*60)
print("For now, modify the adaptive test to use fresh components.")
print("This ensures production and optimizer get identical results.")
print("="*60)