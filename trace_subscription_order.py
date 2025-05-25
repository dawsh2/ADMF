#!/usr/bin/env python3
"""
Trace the exact order of BAR event subscriptions
"""
import re

def trace_bar_subscriptions(log_file):
    """Find all BAR event subscriptions in order"""
    subscriptions = []
    
    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f):
            # Look for subscription messages
            if "subscribed to event type 'BAR'" in line or "Subscribed to BAR events" in line:
                # Extract the component name
                component = "Unknown"
                
                # Try different patterns
                patterns = [
                    r"Handler '([^']+)' subscribed to event type 'BAR'",
                    r"(\w+) '([^']+)' subscribed to BAR events",
                    r"'([^']+)' subscribed to BAR events",
                    r"(\w+) subscribed to BAR events"
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, line)
                    if match:
                        if match.lastindex == 2:
                            component = f"{match.group(1)} '{match.group(2)}'"
                        else:
                            component = match.group(1)
                        break
                
                # Also check the line content for component identification
                if 'Classifier' in line:
                    component = f"Classifier (RegimeDetector)"
                elif 'MAStrategy' in line or 'EnsembleSignalStrategy' in line:
                    component = f"Strategy"
                elif 'Portfolio' in line:
                    component = f"Portfolio"
                elif 'RiskManager' in line:
                    component = f"RiskManager"
                elif 'DataHandler' in line:
                    component = f"DataHandler"
                
                subscriptions.append({
                    'line_num': line_num,
                    'component': component,
                    'line': line.strip()
                })
    
    return subscriptions

def compare_subscription_orders():
    """Compare subscription orders between production and optimizer"""
    
    print("BAR EVENT SUBSCRIPTION ORDER ANALYSIS")
    print("=" * 80)
    
    prod_file = "/Users/daws/ADMF/logs/admf_20250524_002523.log"
    opt_file = "/Users/daws/ADMF/logs/admf_20250523_230532.log"
    
    print("\n1. PRODUCTION SUBSCRIPTION ORDER:")
    print("-" * 80)
    prod_subs = trace_bar_subscriptions(prod_file)
    for i, sub in enumerate(prod_subs[:15]):
        print(f"{i+1:2d}. Line {sub['line_num']:6d}: {sub['component']}")
        print(f"    {sub['line'][:100]}...")
    
    print("\n2. OPTIMIZER SUBSCRIPTION ORDER:")
    print("-" * 80)
    opt_subs = trace_bar_subscriptions(opt_file)
    for i, sub in enumerate(opt_subs[:15]):
        print(f"{i+1:2d}. Line {sub['line_num']:6d}: {sub['component']}")
        print(f"    {sub['line'][:100]}...")
    
    print("\n3. KEY DIFFERENCES:")
    print("-" * 80)
    
    # Compare the order
    print("\nProduction order summary:")
    for i, sub in enumerate(prod_subs[:10]):
        print(f"  {i+1}. {sub['component']}")
    
    print("\nOptimizer order summary:")
    for i, sub in enumerate(opt_subs[:10]):
        print(f"  {i+1}. {sub['component']}")
    
    print("\n4. IMPACT ON EVENT PROCESSING:")
    print("-" * 80)
    print("\nThe EventBus processes subscribers in the order they subscribed.")
    print("If components subscribe in different orders, they process events in different orders.")
    print("\nAt regime boundaries:")
    print("- Production: Strategy may process BAR before RegimeDetector")
    print("- Optimizer: RegimeDetector may process BAR before Strategy")
    print("\nThis creates the race condition where signals use different regimes!")

def main():
    compare_subscription_orders()

if __name__ == "__main__":
    main()