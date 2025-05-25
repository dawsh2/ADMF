#!/usr/bin/env python3
"""
Wrapper to run production with warmup phase matching optimizer behavior.
This temporarily patches the ensemble strategy to handle warmup.
"""

import os
import sys
import subprocess
import json
from datetime import datetime

class WarmupWrapper:
    """Manages the warmup patching and restoration."""
    
    def __init__(self):
        self.strategy_file = "src/strategy/implementations/ensemble_strategy.py"
        self.backup_file = self.strategy_file + ".warmup_backup"
        self.warmup_bars = 798  # Match optimizer's training size
        
    def backup_original(self):
        """Backup the original strategy file."""
        if not os.path.exists(self.backup_file):
            subprocess.run(f"cp {self.strategy_file} {self.backup_file}", shell=True)
            print(f"✓ Backed up original to {self.backup_file}")
        else:
            print(f"✓ Backup already exists at {self.backup_file}")
    
    def restore_original(self):
        """Restore the original strategy file."""
        if os.path.exists(self.backup_file):
            subprocess.run(f"cp {self.backup_file} {self.strategy_file}", shell=True)
            print(f"✓ Restored original from {self.backup_file}")
        else:
            print("⚠ No backup found to restore")
    
    def patch_strategy(self):
        """Add warmup handling to the strategy."""
        print(f"✓ Patching strategy with {self.warmup_bars}-bar warmup...")
        
        with open(self.strategy_file, 'r') as f:
            content = f.read()
        
        # Check if already patched
        if "_warmup_bars" in content:
            print("  Strategy already patched")
            return True
        
        # Find the __init__ method
        init_pattern = "def __init__(self, instance_name: str,"
        init_pos = content.find(init_pattern)
        if init_pos == -1:
            print("  ERROR: Could not find __init__ method")
            return False
        
        # Find where to insert warmup init (end of __init__)
        # Look for the logger.info line near end of init
        logger_pattern = 'self.logger.info(f"EnsembleSignalStrategy \'{self.name}\' initialized'
        logger_pos = content.find(logger_pattern, init_pos)
        if logger_pos == -1:
            print("  ERROR: Could not find init logger line")
            return False
        
        # Find end of that line
        line_end = content.find('\n', logger_pos)
        
        # Insert warmup initialization
        warmup_init = f"""
        
        # === WARMUP PATCH START ===
        self._warmup_bars = {self.warmup_bars}
        self._bars_processed = 0
        self._in_warmup = True
        self.logger.info(f"[WARMUP] Enabled for {{self._warmup_bars}} bars")
        # === WARMUP PATCH END ==="""
        
        content = content[:line_end] + warmup_init + content[line_end:]
        
        # Now patch the _on_bar_event method
        bar_pattern = "def _on_bar_event(self, event: Event):"
        bar_pos = content.find(bar_pattern)
        if bar_pos == -1:
            print("  ERROR: Could not find _on_bar_event method")
            return False
        
        # Find the docstring end or first real code
        method_start = content.find('\n', bar_pos) + 1
        
        # Skip any docstring
        pos = method_start
        in_docstring = False
        docstring_char = None
        
        while pos < len(content):
            if content[pos:pos+3] in ['"""', "'''"]:
                if not in_docstring:
                    in_docstring = True
                    docstring_char = content[pos:pos+3]
                    pos += 3
                elif content[pos:pos+3] == docstring_char:
                    in_docstring = False
                    pos += 3
                    # Skip to next line after docstring
                    newline_pos = content.find('\n', pos)
                    if newline_pos != -1:
                        pos = newline_pos + 1
                    break
                else:
                    pos += 1
            elif in_docstring:
                pos += 1
            else:
                # No docstring, this is where code starts
                break
        
        # Insert warmup check
        indent = "        "  # 8 spaces
        warmup_check = f"""
{indent}# === WARMUP PATCH START ===
{indent}if event.event_type == EventType.BAR and event.data.get('symbol') == self._symbol:
{indent}    self._bars_processed += 1
{indent}    
{indent}    if self._bars_processed <= self._warmup_bars:
{indent}        # Warmup phase - update indicators but don't generate signals
{indent}        if self._bars_processed == 1:
{indent}            self.logger.info(f"[WARMUP] Phase started. Processing {{self._warmup_bars}} training bars...")
{indent}        elif self._bars_processed % 100 == 0:
{indent}            self.logger.info(f"[WARMUP] Progress: {{self._bars_processed}}/{{self._warmup_bars}} bars")
{indent}        elif self._bars_processed == self._warmup_bars:
{indent}            self.logger.info(f"[WARMUP] Complete! Indicators warmed. Starting evaluation...")
{indent}            self._in_warmup = False
{indent}        
{indent}        # Process bar for indicators only
{indent}        bar_data = event.data
{indent}        price = bar_data['close']
{indent}        self._prices.append(price)
{indent}        
{indent}        # Update RSI
{indent}        if hasattr(self, 'rsi_indicator'):
{indent}            self.rsi_indicator.update(price)
{indent}        
{indent}        # Skip rest of processing during warmup
{indent}        return
{indent}# === WARMUP PATCH END ===
"""
        
        content = content[:pos] + warmup_check + content[pos:]
        
        # Write patched file
        with open(self.strategy_file, 'w') as f:
            f.write(content)
        
        print("  Patch applied successfully")
        return True

def create_full_data_config():
    """Create config that processes all data."""
    config = {
        "system": {
            "name": "ADMF-Production-Warmup",
            "version": "0.1.0"
        },
        "logging": {
            "level": "INFO"
        },
        "components": {
            "ensemble_strategy": {
                "symbol": "SPY",
                "short_window_default": 10,
                "long_window_default": 20,
                "ma_rule.weight": 1.0,
                "rsi_indicator": {
                    "period": 14
                },
                "rsi_rule": {
                    "oversold_threshold": 30.0,
                    "overbought_threshold": 70.0,
                    "weight": 0.0
                }
            },
            "data_handler_csv": {
                "csv_file_path": "data/1000_1min.csv",
                "symbol": "SPY",
                "timestamp_column": "timestamp",
                "train_test_split_ratio": 1.0,  # Process ALL data
                "open_column": "Open",
                "high_column": "High", 
                "low_column": "Low",
                "close_column": "Close",
                "volume_column": "Volume"
            },
            "basic_portfolio": {
                "initial_cash": 100000.00
            },
            "basic_risk_manager": {
                "target_trade_quantity": 100
            },
            "simulated_execution_handler": {
                "default_quantity": 100,
                "commission_per_trade": 0.005,
                "commission_type": "per_share",
                "passthrough": False,
                "fill_price_logic": "signal_price"
            }
        }
    }
    
    config_path = "config/config_warmup.yaml"
    with open(config_path, 'w') as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path

def run_with_warmup():
    """Main execution with warmup."""
    print("PRODUCTION RUN WITH WARMUP")
    print("=" * 60)
    
    wrapper = WarmupWrapper()
    
    try:
        # Backup original
        wrapper.backup_original()
        
        # Apply patch
        if not wrapper.patch_strategy():
            print("ERROR: Failed to patch strategy")
            return
        
        # Create config
        config_path = create_full_data_config()
        print(f"✓ Created config: {config_path}")
        
        # Run production
        print("\n" + "-"*60)
        print("Running production with warmup...")
        print("Expected: 798 warmup bars, then 200 test bars")
        print("-"*60 + "\n")
        
        cmd = ["python", "main.py", "--config", config_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return
        
        # Analyze results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        # Find latest log
        log_files = sorted([f for f in os.listdir('logs') if f.endswith('.log')], reverse=True)
        if not log_files:
            print("No log files found")
            return
            
        latest_log = f"logs/{log_files[0]}"
        
        # Check warmup messages
        print("\nWarmup phase:")
        subprocess.run(f"grep '\\[WARMUP\\]' {latest_log} | tail -5", shell=True)
        
        # Count signals
        signal_count = subprocess.check_output(
            f"grep -c '🚨 SIGNAL GENERATED' {latest_log} || echo 0", 
            shell=True, text=True
        ).strip()
        print(f"\nTotal signals generated: {signal_count}")
        
        # Check first signal timestamp
        print("\nFirst signal:")
        subprocess.run(
            f"grep -B1 '🚨 SIGNAL GENERATED' {latest_log} | head -2 | grep -E 'BAR_[0-9]+ \\[|SIGNAL GENERATED'", 
            shell=True
        )
        
        # Check if we got signal at 13:46
        print("\nChecking for signal at 13:46:00...")
        subprocess.run(
            f"grep -B1 -A1 '13:46:00' {latest_log} | grep 'SIGNAL GENERATED' || echo 'No signal at 13:46:00'",
            shell=True
        )
        
        print("\n" + "-"*60)
        print("Expected: 16 signals starting at 2024-03-28 13:46:00")
        
    finally:
        # Always restore original
        print("\n" + "-"*60)
        wrapper.restore_original()
        print("\nDone!")

if __name__ == "__main__":
    run_with_warmup()