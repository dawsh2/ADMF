#!/usr/bin/env python3
"""
Script to run both optimizer and standalone tests with detailed indicator logging
to compare the first 50 bars and identify signal generation differences.
"""
import sys
import os
import subprocess
import json
import time

# Add the project root to Python path
sys.path.append('.')

def run_optimizer_test():
    """Run the optimizer adaptive test and capture output."""
    print("=" * 60)
    print("RUNNING OPTIMIZER ADAPTIVE TEST")
    print("=" * 60)
    
    try:
        from src.strategy.optimization.enhanced_optimizer import EnhancedOptimizer
        
        config_path = 'config/config.yaml'
        # Load the best parameters from regime_optimized_parameters.json
        with open('regime_optimized_parameters.json', 'r') as f:
            data = json.load(f)
        
        # Use the joint optimization results
        best_params = data['best_parameters_on_train']
        
        print(f"Using parameters: {best_params}")
        
        optimizer = EnhancedOptimizer(config_path)
        results = optimizer._run_regime_adaptive_test(best_params)
        
        print(f'\n🔹 OPTIMIZER TEST RESULTS:')
        print(f'   Final Value: {results["final_portfolio_value"]:.2f}')
        print(f'   Total Trades: {results["total_trades"]}')
        
        return results
        
    except Exception as e:
        print(f"❌ Error running optimizer test: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_standalone_test():
    """Run the standalone test and capture output."""
    print("\n" + "=" * 60)
    print("RUNNING STANDALONE TEST")
    print("=" * 60)
    
    try:
        # Run the standalone test with limited bars for faster comparison
        cmd = [sys.executable, 'main.py', '--config', 'config/config_adaptive_production.yaml', '--bars', '100000']
        
        print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print("✅ Standalone test completed successfully")
            
            # Extract final results from output
            lines = result.stdout.split('\n')
            final_value = None
            total_trades = None
            
            for line in lines:
                if 'Final Portfolio Value:' in line:
                    try:
                        final_value = float(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Total Trades:' in line or 'trade count:' in line.lower():
                    try:
                        # Extract number from various formats
                        parts = line.split(':')
                        if len(parts) > 1:
                            total_trades = int(parts[1].strip())
                    except:
                        pass
            
            print(f'\n🔹 STANDALONE TEST RESULTS:')
            print(f'   Final Value: {final_value if final_value else "Not found"}')
            print(f'   Total Trades: {total_trades if total_trades else "Not found"}')
            
            return {'final_value': final_value, 'total_trades': total_trades, 'stdout': result.stdout}
        else:
            print(f"❌ Standalone test failed with return code: {result.returncode}")
            print("STDERR:", result.stderr)
            return None
            
    except Exception as e:
        print(f"❌ Error running standalone test: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_indicator_logs(log_content, source_name):
    """Extract the first 50 bars of indicator data from logs."""
    print(f"\n📊 EXTRACTING INDICATOR DATA FROM {source_name}")
    print("-" * 40)
    
    lines = log_content.split('\n') if isinstance(log_content, str) else []
    indicator_lines = []
    signal_lines = []
    
    for line in lines:
        if '📊 BAR_' in line and 'INDICATORS:' in line:
            indicator_lines.append(line)
        elif '🚨 SIGNAL GENERATED' in line:
            signal_lines.append(line)
    
    print(f"Found {len(indicator_lines)} indicator log lines")
    print(f"Found {len(signal_lines)} signal generation lines")
    
    if indicator_lines:
        print("\nFirst 10 indicator logs:")
        for i, line in enumerate(indicator_lines[:10]):
            print(f"  {i+1:2d}. {line}")
    
    if signal_lines:
        print(f"\nFirst 5 signal generations:")
        for i, line in enumerate(signal_lines[:5]):
            print(f"  {i+1:2d}. {line}")
    
    return indicator_lines, signal_lines

def main():
    """Main function to run both tests and compare results."""
    print("🔍 DEBUGGING SIGNAL GENERATION DIFFERENCES")
    print("This script will run both optimizer and standalone tests")
    print("and compare the first 50 bars of indicator values.\n")
    
    # Change to the project directory
    os.chdir('/Users/daws/ADMF')
    
    # Run optimizer test
    optimizer_results = run_optimizer_test()
    
    # Small delay between tests
    time.sleep(1)
    
    # Run standalone test  
    standalone_results = run_standalone_test()
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    if optimizer_results and standalone_results:
        opt_value = optimizer_results.get('final_portfolio_value', 0)
        opt_trades = optimizer_results.get('total_trades', 0)
        
        sta_value = standalone_results.get('final_value', 0)
        sta_trades = standalone_results.get('total_trades', 0)
        
        print(f"Optimizer Test:  Value={opt_value:.2f}, Trades={opt_trades}")
        print(f"Standalone Test: Value={sta_value:.2f}, Trades={sta_trades}")
        
        if opt_value and sta_value:
            value_diff = abs(opt_value - sta_value)
            print(f"Value Difference: {value_diff:.2f}")
            
        if opt_trades and sta_trades:
            trade_diff = abs(opt_trades - sta_trades)
            print(f"Trade Difference: {trade_diff}")
            
            if trade_diff > 0:
                print("❌ TRADE COUNTS DIFFER - Signal generation is different!")
            else:
                print("✅ Trade counts match")
    else:
        print("❌ Could not compare results - one or both tests failed")
    
    # Try to extract indicator logs from recent log files
    print("\n" + "=" * 60)
    print("SEARCHING FOR INDICATOR LOGS IN RECENT LOG FILES")
    print("=" * 60)
    
    import glob
    log_files = glob.glob('logs/admf_*.log')
    if log_files:
        # Get the most recent log file
        latest_log = max(log_files, key=os.path.getmtime)
        print(f"Reading latest log file: {latest_log}")
        
        try:
            with open(latest_log, 'r') as f:
                log_content = f.read()
            
            extract_indicator_logs(log_content, "LATEST_LOG")
            
        except Exception as e:
            print(f"Error reading log file: {e}")
    else:
        print("No log files found in logs/ directory")

if __name__ == '__main__':
    main()