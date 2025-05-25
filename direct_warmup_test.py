#!/usr/bin/env python3
"""
Direct test of warmup solution by manually editing the strategy.
"""

import os
import subprocess
import shutil

def apply_simple_warmup_patch():
    """Apply a simple warmup patch that we know will work."""
    
    strategy_file = "src/strategy/implementations/ensemble_strategy.py"
    backup_file = strategy_file + ".direct_backup"
    
    # Backup
    shutil.copy(strategy_file, backup_file)
    print(f"Backed up to {backup_file}")
    
    # Read file
    with open(strategy_file, 'r') as f:
        lines = f.readlines()
    
    # Find where to add warmup init
    for i, line in enumerate(lines):
        if "self.logger.info(f\"EnsembleSignalStrategy" in line and "initialized" in line:
            # Add warmup vars after this line
            lines.insert(i + 1, "\n")
            lines.insert(i + 2, "        # WARMUP HANDLING\n")
            lines.insert(i + 3, "        self._warmup_bars = 798\n")
            lines.insert(i + 4, "        self._bars_processed = 0\n")
            lines.insert(i + 5, "        self.logger.info(f\"[WARMUP] Enabled for {self._warmup_bars} bars\")\n")
            break
    
    # Find _on_bar_event method
    for i, line in enumerate(lines):
        if "def _on_bar_event(self, event: Event):" in line:
            # Find where to insert (after docstring if exists)
            j = i + 1
            # Skip docstring
            while j < len(lines) and (lines[j].strip().startswith('"""') or 
                                     lines[j].strip().startswith("'''") or
                                     lines[j].strip() == "" or
                                     '"""' in lines[j] or "'''" in lines[j]):
                j += 1
            
            # Insert warmup check
            warmup_code = '''        # WARMUP CHECK
        if event.event_type == EventType.BAR and event.data.get('symbol') == self._symbol:
            self._bars_processed += 1
            if self._bars_processed <= self._warmup_bars:
                if self._bars_processed == 1:
                    self.logger.info(f"[WARMUP] Starting...")
                elif self._bars_processed == self._warmup_bars:
                    self.logger.info(f"[WARMUP] Complete!")
                # Update indicators
                bar_data = event.data
                self._prices.append(bar_data['close'])
                if hasattr(self, 'rsi_indicator'):
                    self.rsi_indicator.update(bar_data['close'])
                return
        
'''
            lines.insert(j, warmup_code)
            break
    
    # Write file
    with open(strategy_file, 'w') as f:
        f.writelines(lines)
    
    print("Applied warmup patch")
    return backup_file

def test_warmup():
    """Test the warmup solution."""
    print("DIRECT WARMUP TEST")
    print("=" * 60)
    
    backup = None
    try:
        # Apply patch
        backup = apply_simple_warmup_patch()
        
        # Create config for all data
        config = """system:
  name: "ADMF-Direct-Warmup"
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
    train_test_split_ratio: 1.0
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
        
        with open("config/config_direct_warmup.yaml", 'w') as f:
            f.write(config)
        
        # Run
        print("\nRunning with warmup...")
        subprocess.run(["python", "main.py", "--config", "config/config_direct_warmup.yaml"])
        
        # Check results
        print("\n" + "="*60)
        print("Checking results...")
        
        # Find latest log
        import glob
        logs = sorted(glob.glob("logs/admf*.log"), key=os.path.getmtime)
        if logs:
            latest = logs[-1]
            
            # Check warmup
            warmup_count = subprocess.check_output(f"grep -c '\\[WARMUP\\]' {latest} || echo 0", shell=True, text=True).strip()
            print(f"Warmup messages: {warmup_count}")
            
            # Count signals
            signal_count = subprocess.check_output(f"grep -c 'SIGNAL GENERATED' {latest} || echo 0", shell=True, text=True).strip()
            print(f"Total signals: {signal_count}")
            
            # Check first signal
            print("\nFirst signal:")
            subprocess.run(f"grep -m1 -B1 'SIGNAL GENERATED' {latest} || echo 'No signals found'", shell=True)
            
    finally:
        # Restore
        if backup and os.path.exists(backup):
            shutil.copy(backup, "src/strategy/implementations/ensemble_strategy.py")
            os.remove(backup)
            print("\nRestored original file")

if __name__ == "__main__":
    test_warmup()