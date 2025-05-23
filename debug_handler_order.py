#!/usr/bin/env python3
"""Debug handler subscription order"""

def analyze_handler_order(log_file, label):
    """Extract handler subscription order"""
    print(f"\n{label} HANDLER SUBSCRIPTIONS:")
    print("-" * 60)
    
    subscriptions = []
    with open(log_file, 'r') as f:
        for line in f:
            if "subscribed to" in line and ("BAR" in line or "CLASSIFICATION" in line):
                # Extract component name and event types
                if "DEBUG" in line and "Handler" in line:
                    continue  # Skip debug handler messages
                    
                component = None
                if "component." in line:
                    parts = line.split("component.")
                    if len(parts) > 1:
                        component = parts[1].split()[0].strip(" -")
                elif "ensemble_strategy." in line:
                    component = "EnsembleStrategy"
                elif "regime_detector." in line or "MyPrimaryRegimeDetector" in line:
                    component = "RegimeDetector"
                    
                if component:
                    subscriptions.append((component, line.strip()))
                    
    # Show first few subscriptions
    for comp, line in subscriptions[:10]:
        print(f"{comp}: {line[100:200]}...")  # Show part of the line
        
    return subscriptions

# Analyze both logs
prod_subs = analyze_handler_order('/Users/daws/ADMF/logs/admf_20250522_225839.log', "PRODUCTION")
opt_subs = analyze_handler_order('/Users/daws/ADMF/logs/admf_20250522_225752.log', "OPTIMIZATION")

print("\nKEY INSIGHT:")
print("-" * 60)
print("The order of BAR and CLASSIFICATION event handlers can affect timing.")
print("If RegimeDetector processes BAR before Strategy, regime changes happen first.")
print("If Strategy processes BAR before RegimeDetector updates, it uses old regime.")

# Check for timing differences
print("\nCHECKING PROCESSING ORDER:")
print("-" * 60)

# Look for evidence of different processing order
with open('/Users/daws/ADMF/logs/admf_20250522_225839.log', 'r') as f:
    prod_lines = f.readlines()
    
# Find where components process the 16:52 bar
print("\nProduction: Looking for 16:52 bar processing...")
for i, line in enumerate(prod_lines):
    if "2024-04-24 16:52:00" in line and "on_bar" in line:
        print(f"  {line.strip()[:150]}...")
        if i < len(prod_lines) - 1:
            print(f"  {prod_lines[i+1].strip()[:150]}...")
        break