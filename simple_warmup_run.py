#!/usr/bin/env python3
"""
Simple script to run production with warmup by using subprocess.
"""

import subprocess
import os
import sys
import time

def create_warmup_config():
    """Create a config that uses full data (no split)."""
    config_content = """# Config for warmup testing
system:
  name: "ADMF-Warmup-Test"
  version: "0.1.0"

logging:
  level: "INFO"

components:
  ensemble_strategy:
    symbol: "SPY"
    short_window_default: 10
    long_window_default: 20
    ma_rule.weight: 1.0
    rsi_indicator:
      period: 14
    rsi_rule:
      oversold_threshold: 30.0
      overbought_threshold: 70.0
      weight: 0.0

  data_handler_csv:
    csv_file_path: "data/1000_1min.csv"
    symbol: "SPY"
    timestamp_column: "timestamp"
    train_test_split_ratio: 1.0  # Process ALL data
    open_column: "Open"
    high_column: "High"
    low_column: "Low"
    close_column: "Close"
    volume_column: "Volume"

  basic_portfolio:
    initial_cash: 100000.00

  basic_risk_manager:
    target_trade_quantity: 100

  simulated_execution_handler:
    default_quantity: 100
    commission_per_trade: 0.005
    commission_type: "per_share"
    passthrough: false
    fill_price_logic: "signal_price"
"""
    
    with open("config/config_warmup_test.yaml", 'w') as f:
        f.write(config_content)
    
    return "config/config_warmup_test.yaml"

def add_warmup_to_ensemble_strategy():
    """Add warmup logic directly to ensemble strategy file."""
    
    # Read the current ensemble strategy
    with open("src/strategy/implementations/ensemble_strategy.py", 'r') as f:
        content = f.read()
    
    # Check if warmup logic already exists
    if "_warmup_bars" in content:
        print("Warmup logic already present in ensemble strategy")
        return False
    
    # Find where to insert warmup initialization
    init_end = content.find("self.logger.info(f\"EnsembleSignalStrategy")
    if init_end == -1:
        print("Could not find insertion point in __init__")
        return False
    
    # Find the end of that line
    line_end = content.find('\n', init_end)
    
    # Insert warmup variables
    warmup_init = """
        
        # Warmup handling
        self._warmup_bars = 798  # Match optimizer's training size
        self._bars_processed = 0
        self._in_warmup = True
        self.logger.info(f"Warmup mode enabled for {self._warmup_bars} bars")"""
    
    content = content[:line_end] + warmup_init + content[line_end:]
    
    # Find handle_event method
    handle_start = content.find("def handle_event(self, event: Event):")
    if handle_start == -1:
        print("Could not find handle_event method")
        return False
    
    # Find the method body start
    method_body_start = content.find('\n', handle_start) + 1
    indent = "        "  # 8 spaces for method body
    
    # Insert warmup check at beginning of handle_event
    warmup_check = f"""{indent}# Warmup phase handling
{indent}if event.event_type == EventType.BAR and event.data.get('symbol') == self._symbol:
{indent}    self._bars_processed += 1
{indent}    
{indent}    if self._bars_processed <= self._warmup_bars:
{indent}        if self._bars_processed == 1:
{indent}            self.logger.info("[WARMUP] Starting warmup phase...")
{indent}        elif self._bars_processed == self._warmup_bars:
{indent}            self.logger.info("[WARMUP] Complete! Starting evaluation...")
{indent}            self._in_warmup = False
{indent}        
{indent}        # Update indicators during warmup
{indent}        bar_data = event.data
{indent}        price = bar_data['close']
{indent}        self._prices.append(price)
{indent}        
{indent}        if hasattr(self, 'rsi_indicator'):
{indent}            self.rsi_indicator.update(price)
{indent}        
{indent}        # Skip signal generation during warmup
{indent}        return
{indent}
"""
    
    # Find where the original handle_event code starts (after docstring if present)
    # Look for the first non-comment, non-docstring line
    search_pos = method_body_start
    while search_pos < len(content):
        line_start = search_pos
        line_end = content.find('\n', search_pos)
        if line_end == -1:
            line_end = len(content)
        
        line = content[line_start:line_end].strip()
        
        # Skip empty lines, comments, and docstrings
        if line and not line.startswith('#') and not line.startswith('"""') and not line.startswith("'''"):
            # Found the start of actual code
            content = content[:line_start] + warmup_check + content[line_start:]
            break
        
        search_pos = line_end + 1
    
    # Backup original file
    backup_path = "src/strategy/implementations/ensemble_strategy.py.backup"
    if not os.path.exists(backup_path):
        subprocess.run(f"cp src/strategy/implementations/ensemble_strategy.py {backup_path}", shell=True)
        print(f"Backed up original to {backup_path}")
    
    # Write modified version
    with open("src/strategy/implementations/ensemble_strategy.py", 'w') as f:
        f.write(content)
    
    print("Added warmup logic to ensemble strategy")
    return True

def run_production_with_warmup():
    """Run production with warmup."""
    print("TESTING WARMUP SOLUTION")
    print("=" * 60)
    
    # Create config
    config_path = create_warmup_config()
    print(f"Created config: {config_path}")
    
    # Add warmup logic to strategy
    if add_warmup_to_ensemble_strategy():
        print("\nModified ensemble strategy with warmup handling")
    
    # Run production
    print("\nRunning production...")
    print("Expected behavior:")
    print("- Process 798 warmup bars (no signals)")
    print("- Then process 200 test bars (with signals)")
    print("-" * 40)
    
    cmd = ["python", "main.py", "--config", config_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return
    
    # Find latest log
    log_files = sorted([f for f in os.listdir('logs') if f.endswith('.log')], reverse=True)
    if not log_files:
        print("No log files found")
        return
    
    latest_log = f"logs/{log_files[0]}"
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    
    # Check warmup messages
    print("\nWarmup messages:")
    subprocess.run(f"grep '\\[WARMUP\\]' {latest_log}", shell=True)
    
    # Count signals
    signal_count = subprocess.check_output(f"grep -c '🚨 SIGNAL GENERATED' {latest_log} || true", shell=True, text=True).strip()
    print(f"\nTotal signals: {signal_count}")
    
    # Show first few signals with timestamps
    print("\nFirst 3 signals:")
    subprocess.run(f"grep -B1 '🚨 SIGNAL GENERATED' {latest_log} | grep -E 'BAR_|SIGNAL GENERATED' | head -6", shell=True)
    
    # Compare with optimizer
    print("\n" + "-"*40)
    print("Expected optimizer signals:")
    print("1. BUY at 2024-03-28 13:46:00")
    print("2. SELL at 2024-03-28 14:00:00")
    print("... (16 total)")
    
    # Restore original file
    print("\n" + "-"*40)
    print("Restoring original ensemble strategy...")
    subprocess.run("cp src/strategy/implementations/ensemble_strategy.py.backup src/strategy/implementations/ensemble_strategy.py", shell=True)
    print("Done!")

if __name__ == "__main__":
    run_production_with_warmup()