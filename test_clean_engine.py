#!/usr/bin/env python3
"""
Test script for CleanBacktestEngine with train/test split.
"""

import argparse
import logging
from src.core.application_launcher import ApplicationLauncher

def main():
    """Test the CleanBacktestEngine implementation."""
    
    # Setup arguments
    parser = argparse.ArgumentParser(description='Test CleanBacktestEngine')
    parser.add_argument('--config', default='config/config_test_clean_engine.yaml', 
                       help='Configuration file path')
    parser.add_argument('--bars', type=int, default=1000,
                       help='Number of bars to use for testing')
    parser.add_argument('--log-level', default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting CleanBacktestEngine test...")
    logger.info(f"Config: {args.config}")
    logger.info(f"Bars limit: {args.bars}")
    
    # Launch the application
    launcher = ApplicationLauncher()
    
    # Convert args to list format expected by launcher
    launch_args = [
        '--config', args.config,
        '--bars', str(args.bars),
        '--log-level', args.log_level
    ]
    
    try:
        result = launcher.launch(launch_args)
        logger.info(f"Test completed with result: {result}")
        return result
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())