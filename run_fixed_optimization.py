#!/usr/bin/env python3
"""
Run optimization with the fix for RegimeAdaptiveStrategy.

This script:
1. Applies the fix to the EnhancedOptimizer
2. Runs the main.py script with the --optimize flag
"""

import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_fix():
    """Apply the fix for RegimeAdaptiveStrategy"""
    logger.info("Applying fix for RegimeAdaptiveStrategy...")
    try:
        # Import the fix module to apply the monkey patch
        from fix_regime_adaptive_strategy import fix_enhanced_optimizer
        fix_enhanced_optimizer()
        logger.info("Fix applied successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to apply fix: {e}", exc_info=True)
        return False

def run_optimization(bars=1000):
    """Run main.py with the --optimize flag"""
    cmd = [
        sys.executable,  # Current Python executable
        "main.py",
        "--config", "config/config.yaml",
        "--optimize",
        "--bars", str(bars),
        "--log-level", "DEBUG"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run the command and stream output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Stream output
    for line in process.stdout:
        print(line, end='')
    
    # Wait for the process to complete
    return_code = process.wait()
    
    if return_code != 0:
        logger.error(f"Optimization failed with return code {return_code}")
        return False
        
    logger.info("Optimization completed successfully")
    return True

if __name__ == "__main__":
    # Get number of bars from command line arguments
    bars = 1000
    if len(sys.argv) > 1:
        try:
            bars = int(sys.argv[1])
        except ValueError:
            logger.warning(f"Invalid number of bars: {sys.argv[1]}. Using default: 1000")
    
    # Apply the fix
    if apply_fix():
        # Run the optimization
        success = run_optimization(bars)
        sys.exit(0 if success else 1)
    else:
        logger.error("Failed to apply fix. Exiting.")
        sys.exit(1)