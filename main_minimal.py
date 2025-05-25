#!/usr/bin/env python3
"""
Minimal main.py - only captures and routes command line arguments.

The actual logic is delegated to the ApplicationRunner which uses
Bootstrap and configuration to determine what to run.
"""

import sys
from src.core.application_runner import ApplicationRunner


def main():
    """
    Main entry point - just passes command line args to the application runner.
    
    This is intentionally minimal. All logic including:
    - Argument parsing
    - Configuration loading
    - Run mode determination
    - Component setup
    - Execution
    
    Is handled by ApplicationRunner and Bootstrap.
    """
    try:
        # Create runner with command line args
        runner = ApplicationRunner(sys.argv[1:])
        
        # Run the application
        exit_code = runner.run()
        
        # Exit with appropriate code
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()