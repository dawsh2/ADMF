#!/usr/bin/env python3
"""
Override the train/test split handling to process all data.
"""

import os
import shutil
import subprocess

def patch_csv_handler_split():
    """Patch CSV handler to override split behavior."""
    
    handler_file = "src/data/csv_data_handler.py"
    backup_file = handler_file + ".split_backup"
    
    # Backup
    shutil.copy(handler_file, backup_file)
    print(f"Backed up to {backup_file}")
    
    # Read file
    with open(handler_file, 'r') as f:
        content = f.read()
    
    # Find where test data is set
    # Look for: self._active_data = self._test_data
    old_line = "self._active_data = self._test_data"
    new_line = """# WARMUP OVERRIDE - Process ALL data
        self._active_data = self._data  # Use all data instead of just test
        self.logger.info(f"[WARMUP OVERRIDE] Using ALL {len(self._data)} bars instead of {len(self._test_data)} test bars")"""
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        print("Patched data handler to use all data")
    else:
        print("Could not find line to patch")
        return None
    
    # Write patched file
    with open(handler_file, 'w') as f:
        f.write(content)
    
    return backup_file

def patch_strategy_to_skip_early_signals():
    """Patch strategy to skip signals during first 798 bars."""
    
    strategy_file = "src/strategy/implementations/ensemble_strategy.py"
    backup_file = strategy_file + ".skip_backup"
    
    shutil.copy(strategy_file, backup_file)
    
    with open(strategy_file, 'r') as f:
        content = f.read()
    
    # Add bar counter to __init__
    init_end = content.find("self.logger.info(f\"EnsembleSignalStrategy")
    if init_end > 0:
        line_end = content.find('\n', init_end)
        bar_counter_init = """
        
        # WARMUP: Skip signals for first 798 bars
        self._bar_counter = 0
        self._warmup_complete = False"""
        content = content[:line_end] + bar_counter_init + content[line_end:]
    
    # Add check to _on_bar_event
    bar_method = content.find("def _on_bar_event(self, event: Event):")
    if bar_method > 0:
        # Find first real code line after method def
        pos = content.find('\n', bar_method) + 1
        # Skip any docstring
        while pos < len(content):
            line = content[pos:content.find('\n', pos)].strip()
            if line and not line.startswith('"""') and not line.startswith('#'):
                break
            pos = content.find('\n', pos) + 1
        
        signal_skip = """        # WARMUP: Count bars and skip early signals
        if event.event_type == EventType.BAR and event.data.get('symbol') == self._symbol:
            self._bar_counter += 1
            if self._bar_counter == 798:
                self.logger.info("[WARMUP] Complete after 798 bars! Starting signal generation...")
                self._warmup_complete = True
            elif self._bar_counter < 798:
                # Process bar but skip rest of method to prevent signals
                bar_data = event.data
                price = bar_data['close']
                self._prices.append(price)
                if hasattr(self, 'rsi_indicator'):
                    self.rsi_indicator.update(price)
                return
        
"""
        content = content[:pos] + signal_skip + content[pos:]
    
    with open(strategy_file, 'w') as f:
        f.write(content)
    
    print("Patched strategy to skip first 798 bars")
    return backup_file

def test_override():
    """Test the override approach."""
    print("TESTING SPLIT OVERRIDE APPROACH")
    print("=" * 60)
    
    handler_backup = None
    strategy_backup = None
    
    try:
        # Apply patches
        handler_backup = patch_csv_handler_split()
        strategy_backup = patch_strategy_to_skip_early_signals()
        
        if not handler_backup or not strategy_backup:
            print("Failed to apply patches")
            return
        
        # Run with standard config
        print("\nRunning with override patches...")
        print("Expected behavior:")
        print("- Process all 998 bars") 
        print("- Skip signals for bars 1-798")
        print("- Generate signals for bars 799-998")
        print("-" * 60)
        
        result = subprocess.run(
            ["python", "main.py", "--config", "config/config.yaml"],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return
        
        # Check results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        import glob
        logs = sorted(glob.glob("logs/admf*.log"), key=os.path.getmtime)
        if logs:
            latest = logs[-1]
            
            # Check override message
            override_msg = subprocess.check_output(
                f"grep -c 'WARMUP OVERRIDE' {latest} || echo 0",
                shell=True, text=True
            ).strip()
            print(f"Override applied: {'Yes' if override_msg != '0' else 'No'}")
            
            # Check warmup complete
            warmup_complete = subprocess.check_output(
                f"grep -c 'WARMUP.*Complete' {latest} || echo 0",
                shell=True, text=True
            ).strip()
            print(f"Warmup completed: {'Yes' if warmup_complete != '0' else 'No'}")
            
            # Count signals
            signal_count = subprocess.check_output(
                f"grep -c 'SIGNAL GENERATED' {latest} || echo 0",
                shell=True, text=True
            ).strip()
            print(f"Total signals: {signal_count}")
            
            # Check first signal
            print("\nFirst signal:")
            subprocess.run(
                f"grep -m1 -B1 'SIGNAL GENERATED' {latest} | grep -E 'BAR_|SIGNAL' || echo 'No signals'",
                shell=True
            )
            
            # Look for 13:46 timestamp
            print("\nChecking 13:46:00 timestamp:")
            subprocess.run(
                f"grep -C2 '13:46:00' {latest} | grep -E 'BAR_|SIGNAL' | head -5",
                shell=True
            )
            
    finally:
        # Restore files
        if handler_backup:
            shutil.copy(handler_backup, "src/data/csv_data_handler.py")
            os.remove(handler_backup)
            print("\nRestored data handler")
            
        if strategy_backup:
            shutil.copy(strategy_backup, "src/strategy/implementations/ensemble_strategy.py")
            os.remove(strategy_backup)
            print("Restored strategy")

if __name__ == "__main__":
    test_override()