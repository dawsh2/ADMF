#!/usr/bin/env python3
"""
Validate performance metrics and debug any issues.
"""

import json
import os

def validate_performance():
    """Run validation checks based on the performance debug checklist."""
    
    print("="*80)
    print("PERFORMANCE VALIDATION")
    print("="*80)
    
    # 1. Check parameter file
    print("\n1. PARAMETER FILE CHECK:")
    print("-"*40)
    
    param_file = "test_regime_parameters.json"
    if os.path.exists(param_file):
        with open(param_file, 'r') as f:
            params = json.load(f)
        print(f"✅ {param_file} exists")
        print(f"   Timestamp: {params.get('timestamp', 'unknown')}")
        print(f"   Regimes with parameters: {list(params.get('regimes', {}).keys())}")
        
        # Show a sample of parameters
        default_params = params.get('regimes', {}).get('default', {})
        if default_params:
            print("\n   Sample DEFAULT parameters:")
            for key, value in list(default_params.items())[:5]:
                print(f"     {key}: {value}")
    else:
        print(f"❌ {param_file} NOT FOUND!")
        print("   Run optimization to generate parameters")
    
    # 2. Check for recent optimization results
    print("\n2. OPTIMIZATION RESULTS CHECK:")
    print("-"*40)
    
    opt_dir = "optimization_results"
    if os.path.exists(opt_dir):
        files = sorted([f for f in os.listdir(opt_dir) if f.endswith('.json')])
        if files:
            print(f"✅ Found {len(files)} optimization results")
            latest = files[-1]
            print(f"   Latest: {latest}")
            
            # Check if results are recent
            with open(os.path.join(opt_dir, latest), 'r') as f:
                result = json.load(f)
            print(f"   Best score: {result.get('best_score', 'unknown')}")
        else:
            print("❌ No optimization results found")
    else:
        print("❌ optimization_results directory not found")
    
    # 3. Create test commands
    print("\n3. TEST COMMANDS:")
    print("-"*40)
    print("Run these commands to test different aspects:\n")
    
    print("# Test with small dataset (quick check):")
    print("python3 main_ultimate.py --config config/test_ensemble_optimization.yaml --bars 1000 --log-level INFO\n")
    
    print("# Test with test dataset (uses optimized parameters):")
    print("python3 main_ultimate.py --config config/test_ensemble_optimization.yaml --dataset test --bars 5000 --log-level INFO\n")
    
    print("# Test with debug logging (see all decisions):")
    print("python3 main_ultimate.py --config config/test_ensemble_optimization.yaml --bars 1000 --log-level DEBUG 2>&1 | tee debug_run.log\n")
    
    # 4. Check recent logs
    print("4. RECENT LOG CHECK:")
    print("-"*40)
    
    log_dir = "logs"
    if os.path.exists(log_dir):
        logs = sorted([f for f in os.listdir(log_dir) if f.startswith('admf_') and f.endswith('.log')])
        if logs:
            latest_log = logs[-1]
            print(f"✅ Latest log: {latest_log}")
            
            # Quick check for issues in latest log
            log_path = os.path.join(log_dir, latest_log)
            with open(log_path, 'r') as f:
                content = f.read(10000)  # First 10KB
                
            issues = []
            if "Number of Trades: 0" in content:
                issues.append("Zero trades reported")
            if "ERROR" in content:
                issues.append("Errors found in log")
            if "Sharpe Ratio: -" not in content and "Total Return: -" in content:
                issues.append("Negative return but possibly positive Sharpe")
                
            if issues:
                print(f"⚠️  Potential issues found:")
                for issue in issues:
                    print(f"   - {issue}")
            else:
                print("✅ No obvious issues in log header")
    else:
        print("❌ logs directory not found")
    
    # 5. Performance metrics to watch
    print("\n5. KEY METRICS TO MONITOR:")
    print("-"*40)
    print("When running tests, check for:")
    print("- Total Return: Should match (Final - Initial) / Initial")
    print("- Number of Trades: Should be > 0 and match regime totals")
    print("- Sharpe Ratio: Should be negative when returns are negative")
    print("- Regime Performance: Each regime should show trade counts")
    print("- Commission: Should be deducted ($0.005 per trade default)")
    
    # 6. Debug suggestions
    print("\n6. DEBUGGING TIPS:")
    print("-"*40)
    print("If performance is poor:")
    print("1. Check if parameters are being loaded (look for 'LOADING REGIME ADAPTIVE')")
    print("2. Verify regime changes occur (look for 'REGIME CHANGE DETECTED')")
    print("3. Count signals vs trades (signals should lead to orders)")
    print("4. Check portfolio value progression (should change with trades)")
    print("5. Verify no resets during trading (only at start)")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Run a test with the commands above")
    print("2. Check if trades are executed (non-zero count)")
    print("3. Verify Sharpe ratio sign matches return sign")
    print("4. Look for regime switching in test runs")
    print("5. Compare performance with earlier good results")

if __name__ == "__main__":
    validate_performance()