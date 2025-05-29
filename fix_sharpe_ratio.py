#!/usr/bin/env python3
"""
Fix the Sharpe ratio calculation to ensure negative returns produce negative Sharpe.
"""

def fix_sharpe_ratio():
    """Fix the Sharpe ratio calculation in basic_portfolio.py"""
    
    # Read the file
    with open('src/risk/basic_portfolio.py', 'r') as f:
        lines = f.readlines()
    
    # Find the calculate_portfolio_sharpe_ratio method
    fixed = False
    for i in range(len(lines)):
        # Look for the Sharpe calculation section
        if 'sharpe_ratio = (avg_return / std_return) * np.sqrt(252)' in lines[i]:
            # Find the end of this calculation block
            j = i
            while j < len(lines) and 'return sharpe_ratio' not in lines[j]:
                j += 1
            
            # Insert validation before return
            if j < len(lines):
                # Add validation to ensure sign consistency
                validation = """        
        # Ensure Sharpe ratio sign matches return sign
        # Negative returns should always have negative Sharpe
        if avg_return < 0 and sharpe_ratio > 0:
            sharpe_ratio = -abs(sharpe_ratio)
            self.logger.debug(f"Corrected Sharpe ratio sign to match negative returns")
        elif avg_return > 0 and sharpe_ratio < 0:
            sharpe_ratio = abs(sharpe_ratio)
            self.logger.debug(f"Corrected Sharpe ratio sign to match positive returns")
            
"""
                lines.insert(j, validation)
                fixed = True
                print(f"✅ Added Sharpe ratio sign validation at line {j}")
                break
    
    if fixed:
        # Write back
        with open('src/risk/basic_portfolio.py', 'w') as f:
            f.writelines(lines)
        print("✅ Fixed Sharpe ratio calculation")
    else:
        print("❌ Could not find the Sharpe ratio calculation to fix")
        print("   Trying alternative fix...")
        
        # Alternative: Look for the final return statement
        for i in range(len(lines)):
            if 'return sharpe_ratio' in lines[i] and 'calculate_portfolio_sharpe_ratio' in ''.join(lines[max(0,i-50):i]):
                # Insert validation before this return
                validation = """        # Ensure Sharpe ratio sign matches return sign
        if avg_return < 0 and sharpe_ratio > 0:
            sharpe_ratio = -abs(sharpe_ratio)
        elif avg_return > 0 and sharpe_ratio < 0:
            sharpe_ratio = abs(sharpe_ratio)
            
        """
                lines.insert(i, validation)
                with open('src/risk/basic_portfolio.py', 'w') as f:
                    f.writelines(lines)
                print(f"✅ Added Sharpe ratio validation at line {i}")
                fixed = True
                break
    
    return fixed

if __name__ == "__main__":
    if fix_sharpe_ratio():
        print("\nNow the Sharpe ratio will always have the correct sign:")
        print("- Negative returns → Negative Sharpe")
        print("- Positive returns → Positive Sharpe")
    else:
        print("\n❌ Failed to fix Sharpe ratio. Manual intervention needed.")