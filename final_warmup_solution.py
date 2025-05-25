#!/usr/bin/env python3
"""
Final warmup solution - Override the data handler behavior to process all data
but only generate signals after warmup.
"""

import os
import subprocess
import shutil
import pandas as pd

def create_custom_data_handler():
    """Create a custom data handler that processes all bars but marks warmup phase."""
    
    handler_code = '''# Custom CSV handler with warmup support
from src.data.csv_data_handler import CSVDataHandler as BaseCSVDataHandler
from src.core.event import Event, EventType
import pandas as pd

class WarmupCSVDataHandler(BaseCSVDataHandler):
    """CSV handler that processes all data but marks warmup bars."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._warmup_bars = 798
        self._bar_count = 0
        
    def _publish_bar_events(self):
        """Override to mark warmup bars."""
        if self._active_data is None or self._active_data.empty:
            self.logger.warning(f"No active data to publish for '{self.name}'.")
            return

        # Process ALL data
        all_data = pd.read_csv(self._csv_file_path)
        all_data[self._timestamp_column] = pd.to_datetime(all_data[self._timestamp_column])
        all_data.set_index(self._timestamp_column, inplace=True)
        
        self.logger.info(f"Processing {len(all_data)} total bars with {self._warmup_bars} warmup bars")
        
        for timestamp, row in all_data.iterrows():
            self._bar_count += 1
            
            bar_data = self._prepare_bar_data(timestamp, row)
            
            # Mark if this is a warmup bar
            if self._bar_count <= self._warmup_bars:
                bar_data['is_warmup'] = True
                if self._bar_count == 1:
                    self.logger.info("[DATA] Starting warmup phase...")
                elif self._bar_count == self._warmup_bars:
                    self.logger.info("[DATA] Warmup complete! Starting evaluation...")
            else:
                bar_data['is_warmup'] = False
            
            bar_event = Event(event_type=EventType.BAR, data=bar_data)
            self._event_bus.publish(bar_event)
            
        self.logger.info(f"Finished publishing {self._bar_count} BAR events")

# Replace the original handler
CSVDataHandler = WarmupCSVDataHandler
'''
    
    # Write custom handler
    with open("src/data/warmup_csv_handler.py", 'w') as f:
        f.write(handler_code)
    
    print("Created custom warmup data handler")

def patch_main_to_use_custom_handler():
    """Patch main.py to import our custom handler."""
    
    main_file = "main.py"
    backup_file = main_file + ".warmup_backup"
    
    shutil.copy(main_file, backup_file)
    
    with open(main_file, 'r') as f:
        content = f.read()
    
    # Replace the import
    old_import = "from src.data.csv_data_handler import CSVDataHandler"
    new_import = "from src.data.warmup_csv_handler import CSVDataHandler"
    
    content = content.replace(old_import, new_import)
    
    with open(main_file, 'w') as f:
        f.write(content)
    
    print("Patched main.py to use warmup handler")
    return backup_file

def patch_strategy_for_warmup():
    """Patch strategy to handle warmup bars."""
    
    strategy_file = "src/strategy/implementations/ensemble_strategy.py"
    backup_file = strategy_file + ".final_backup"
    
    shutil.copy(strategy_file, backup_file)
    
    with open(strategy_file, 'r') as f:
        lines = f.readlines()
    
    # Find _on_bar_event
    for i, line in enumerate(lines):
        if "def _on_bar_event(self, event: Event):" in line:
            # Insert warmup check after method signature
            j = i + 1
            # Skip any docstring
            while j < len(lines) and (not lines[j].strip() or '"""' in lines[j] or "'''" in lines[j]):
                j += 1
            
            warmup_check = '''        # Check for warmup bars
        if event.event_type == EventType.BAR and event.data.get('symbol') == self._symbol:
            if event.data.get('is_warmup', False):
                # Process bar for indicator updates only
                bar_data = event.data
                price = bar_data['close']
                self._prices.append(price)
                if hasattr(self, 'rsi_indicator'):
                    self.rsi_indicator.update(price)
                # Skip signal generation
                return
        
'''
            lines.insert(j, warmup_check)
            break
    
    with open(strategy_file, 'w') as f:
        f.writelines(lines)
    
    print("Patched strategy to handle warmup")
    return backup_file

def run_final_test():
    """Run the final warmup test."""
    print("FINAL WARMUP TEST")
    print("=" * 60)
    
    main_backup = None
    strategy_backup = None
    
    try:
        # Create custom handler
        create_custom_data_handler()
        
        # Patch files
        main_backup = patch_main_to_use_custom_handler()
        strategy_backup = patch_strategy_for_warmup()
        
        # Create config (with valid split ratio)
        config = """system:
  name: "ADMF-Final-Warmup"
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
    train_test_split_ratio: 0.8  # Valid ratio (handler will ignore it anyway)
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
        
        with open("config/config_final_warmup.yaml", 'w') as f:
            f.write(config)
        
        # Run
        print("\nRunning with warmup data handler...")
        print("Expected: Process 998 bars total")
        print("- Bars 1-798: Warmup (no signals)")  
        print("- Bars 799-998: Evaluation (signals)")
        print("-" * 60)
        
        result = subprocess.run(
            ["python", "main.py", "--config", "config/config_final_warmup.yaml"],
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
            
            # Check data handler messages
            print("\nData handler messages:")
            subprocess.run(f"grep '\\[DATA\\]' {latest}", shell=True)
            
            # Count signals
            signal_count = subprocess.check_output(
                f"grep -c 'SIGNAL GENERATED' {latest} || echo 0", 
                shell=True, text=True
            ).strip()
            print(f"\nTotal signals: {signal_count}")
            
            # First signal with timestamp
            print("\nFirst signal:")
            subprocess.run(
                f"grep -m1 -B1 'SIGNAL GENERATED' {latest} | grep -E 'BAR_|SIGNAL' || echo 'No signals'",
                shell=True
            )
            
            # Check for 13:46 signal
            print("\nChecking for 13:46:00 signal:")
            subprocess.run(
                f"grep -B1 -A1 '13:46:00' {latest} | grep 'SIGNAL' || echo 'No signal at 13:46:00'",
                shell=True
            )
            
    finally:
        # Restore files
        if main_backup:
            shutil.copy(main_backup, "main.py")
            os.remove(main_backup)
        if strategy_backup:
            shutil.copy(strategy_backup, "src/strategy/implementations/ensemble_strategy.py")
            os.remove(strategy_backup)
        if os.path.exists("src/data/warmup_csv_handler.py"):
            os.remove("src/data/warmup_csv_handler.py")
        
        print("\nRestored all files")

if __name__ == "__main__":
    run_final_test()