#!/usr/bin/env python3
"""
Ultimate minimal main.py - truly just an entry point.

All logic is delegated to ApplicationLauncher which sets up
Bootstrap and ensures AppRunner component handles the application flow.
"""

import sys
from src.core.application_launcher import ApplicationLauncher


def main():
    """
    Absolute minimal entry point.
    
    Captures command line arguments and passes them to the launcher.
    That's it. No parsing, no logic, just forwarding.
    """
    # sys.argv[1:] captures all command line arguments after the script name
    launcher = ApplicationLauncher(sys.argv[1:])
    sys.exit(launcher.run())


if __name__ == "__main__":
    main()