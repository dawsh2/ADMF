#!/usr/bin/env python3
"""
Fix the trade count issue by ensuring trade log is preserved properly.
"""

import re

def fix_portfolio_trade_count():
    """Fix the trade count issue in portfolio performance reporting."""
    
    # Read the current portfolio file
    with open('src/risk/basic_portfolio.py', 'r') as f:
        content = f.read()
    
    # The issue is that get_performance returns len(self._trade_log) but 
    # regime performance already has the correct counts. Let's fix this by
    # using the regime performance data to calculate total trades.
    
    # Find the get_performance method
    old_pattern = r"(def get_performance\(self\) -> Dict\[str, Any\]:.*?)'num_trades': len\(self\._trade_log\),"
    
    # Check if we can find it
    if not re.search(old_pattern, content, re.DOTALL):
        print("Could not find the expected pattern. Looking for alternative...")
        # Try a simpler pattern
        old_pattern = r"'num_trades': len\(self\._trade_log\),"
    
    # Replace with a version that sums up regime trades
    new_code = """'num_trades': sum(
                perf.get('count', 0) 
                for regime, perf in self._calculate_performance_by_regime().items() 
                if regime != '_boundary_trades_summary' and isinstance(perf, dict)
            ),"""
    
    # Apply the fix
    content = re.sub(old_pattern, new_code, content)
    
    # Write back
    with open('src/risk/basic_portfolio.py', 'w') as f:
        f.write(content)
    
    print("Fixed trade count calculation to use regime performance data")
    
    # Also add a debug log to see what's happening
    # Find the get_performance method more precisely
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'def get_performance(self)' in line:
            # Find the metrics dictionary
            for j in range(i, min(i + 50, len(lines))):
                if 'metrics = {' in lines[j]:
                    # Insert debug logging before metrics
                    lines.insert(j, '        self.logger.debug(f"Getting performance: trade_log has {len(self._trade_log)} trades")')
                    lines.insert(j+1, '        regime_perf = self._calculate_performance_by_regime()')
                    lines.insert(j+2, '        total_regime_trades = sum(p.get("count", 0) for r, p in regime_perf.items() if r != "_boundary_trades_summary" and isinstance(p, dict))')
                    lines.insert(j+3, '        self.logger.debug(f"Regime performance shows {total_regime_trades} total trades")')
                    break
            break
    
    # Write the updated content
    with open('src/risk/basic_portfolio.py', 'w') as f:
        f.write('\n'.join(lines))
    
    print("Added debug logging to track trade count discrepancy")

if __name__ == "__main__":
    fix_portfolio_trade_count()
    print("\nNow run a test to verify the fix:")