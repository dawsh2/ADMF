#!/usr/bin/env python3
"""
Final working solution - patch main.py to use full dataset and add warmup to strategy.
"""

import os
import shutil
import subprocess

def patch_main_for_full_data():
    """Patch main.py to use full dataset instead of test."""
    
    main_file = "main.py"
    backup_file = main_file + ".final_backup"
    
    shutil.copy(main_file, backup_file)
    
    with open(main_file, 'r') as f:
        content = f.read()
    
    # Find and replace the dataset selection
    old_code = 'data_handler.set_active_dataset("test")'
    new_code = '''# WARMUP PATCH - Use full dataset for warmup
                data_handler.set_active_dataset("full")
                logger.info("[WARMUP PATCH] Using FULL dataset for warmup + evaluation")'''
    
    content = content.replace(old_code, new_code)
    
    with open(main_file, 'w') as f:
        f.write(content)
    
    print("Patched main.py to use full dataset")
    return backup_file

def patch_strategy_with_warmup():
    """Add proper warmup handling to strategy."""
    
    strategy_file = "src/strategy/implementations/ensemble_strategy.py"
    backup_file = strategy_file + ".final_backup"
    
    shutil.copy(strategy_file, backup_file)
    
    with open(strategy_file, 'r') as f:
        lines = f.readlines()
    
    # Add warmup counter to __init__
    for i, line in enumerate(lines):
        if "self.logger.info(f\"EnsembleSignalStrategy" in line and "initialized" in line:
            lines.insert(i + 1, "\n")
            lines.insert(i + 2, "        # Warmup handling\n")
            lines.insert(i + 3, "        self._warmup_bars = 798  # Match optimizer training size\n")
            lines.insert(i + 4, "        self._bar_count = 0\n")
            lines.insert(i + 5, "        self._warmup_complete = False\n")
            break
    
    # Add warmup check to _on_bar_event
    for i, line in enumerate(lines):
        if "def _on_bar_event(self, event: Event):" in line:
            # Find first code line after method signature
            j = i + 1
            while j < len(lines) and (lines[j].strip() == "" or 
                                     lines[j].strip().startswith('"""') or 
                                     '"""' in lines[j]):
                j += 1
            
            warmup_code = '''        # Warmup check
        if event.event_type == EventType.BAR and event.data.get('symbol') == self._symbol:
            self._bar_count += 1
            
            if not self._warmup_complete:
                if self._bar_count < self._warmup_bars:
                    # Update indicators but don't generate signals
                    bar_data = event.data
                    price = bar_data['close']
                    self._prices.append(price)
                    if hasattr(self, 'rsi_indicator'):
                        self.rsi_indicator.update(price)
                    
                    if self._bar_count % 100 == 0:
                        self.logger.info(f"[WARMUP] Progress: {self._bar_count}/{self._warmup_bars} bars")
                    return
                elif self._bar_count == self._warmup_bars:
                    self.logger.info(f"[WARMUP] Complete! Starting signal generation at bar {self._bar_count + 1}")
                    self._warmup_complete = True
        
'''
            lines.insert(j, warmup_code)
            break
    
    with open(strategy_file, 'w') as f:
        f.writelines(lines)
    
    print("Patched strategy with warmup handling")
    return backup_file

def run_final_test():
    """Run the final warmup test."""
    print("FINAL WARMUP SOLUTION TEST")
    print("=" * 60)
    
    main_backup = None
    strategy_backup = None
    
    try:
        # Apply patches
        main_backup = patch_main_for_full_data()
        strategy_backup = patch_strategy_with_warmup()
        
        # Run with standard config (but it will use full data now)
        print("\nRunning with warmup patches...")
        print("Expected behavior:")
        print("- Process ALL 998 bars")
        print("- Bars 1-798: Warmup (update indicators, no signals)")
        print("- Bars 799-998: Normal evaluation (signals)")
        print("-" * 60)
        
        result = subprocess.run(
            ["python", "main.py", "--config", "config/config.yaml"],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return
        
        # Analyze results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        import glob
        logs = sorted(glob.glob("logs/admf*.log"), key=os.path.getmtime)
        if logs:
            latest = logs[-1]
            
            # Check warmup messages
            print("\nWarmup messages:")
            subprocess.run(f"grep '\\[WARMUP' {latest} | tail -5", shell=True)
            
            # Count signals
            signal_count = subprocess.check_output(
                f"grep -c 'SIGNAL GENERATED' {latest} || echo 0",
                shell=True, text=True
            ).strip()
            print(f"\nTotal signals: {signal_count}")
            
            # First signal details
            print("\nFirst signal details:")
            subprocess.run(
                f"grep -m1 -B2 'SIGNAL GENERATED' {latest} | grep -E 'BAR_|timestamp|SIGNAL'",
                shell=True
            )
            
            # Check for 13:46:00
            print("\nChecking for 13:46:00 signal:")
            subprocess.run(
                f"grep -B2 -A2 '13:46:00' {latest} | grep -C3 'SIGNAL' || echo 'No signal found at 13:46:00'",
                shell=True
            )
            
            # Signal timestamps
            print("\nAll signal timestamps:")
            subprocess.run(
                f"grep -B1 'SIGNAL GENERATED' {latest} | grep 'BAR_' | grep -o '\\[[0-9][0-9]:[0-9][0-9]:[0-9][0-9]' | head -5",
                shell=True
            )
            
    finally:
        # Restore files
        if main_backup:
            shutil.copy(main_backup, "main.py")
            os.remove(main_backup)
            print("\nRestored main.py")
            
        if strategy_backup:
            shutil.copy(strategy_backup, "src/strategy/implementations/ensemble_strategy.py")
            os.remove(strategy_backup)
            print("Restored ensemble_strategy.py")

if __name__ == "__main__":
    run_final_test()