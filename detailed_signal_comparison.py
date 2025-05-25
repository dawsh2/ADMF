#!/usr/bin/env python3
"""
Detailed comparison of optimizer vs validation signals with timestamps.
"""

# Optimizer signals with timestamps from the log
optimizer_signals_detailed = [
    ("13:46:00", 523.4800, 1, "First bar - immediate crossover"),
    ("13:48:00", 523.3900, -1, "Early signal"), 
    ("14:05:00", 523.4700, 1, "Matches our bar 818"),
    ("14:14:00", 523.2000, -1, "Not found in validation"),
    ("14:30:00", 523.2300, 1, "Matches our bar 843"),
    ("15:07:00", 523.2500, -1, "Matches our bar 880"),
    ("15:16:00", 523.5000, 1, "Extra signal"),
    ("15:18:00", 523.5000, -1, "Not in validation"),
    ("15:53:00", 523.5600, 1, "Matches our bar 926"),
    ("15:38:00", 523.4250, -1, "MA equal case"),
    ("15:53:00", 523.5600, 1, "Duplicate?"),
    ("16:06:00", 523.2490, -1, "Matches our bar 939"),
    ("16:49:00", 523.2000, 1, "Matches our bar 982"),
    ("17:01:00", 523.1450, -1, "Matches our bar 994"),
    ("17:02:00", 523.2200, 1, "Matches our bar 995"),
    ("17:04:00", 523.2440, -1, "Matches our bar 997"),
]

# Our validation signals
our_signals_detailed = [
    ("Bar 814", "14:01:00", 523.49, -1),
    ("Bar 818", "14:05:00", 523.47, 1),
    ("Bar 825", "14:12:00", 523.05, -1),
    ("Bar 843", "14:30:00", 523.23, 1),
    ("Bar 880", "15:07:00", 523.25, -1),
    ("Bar 892", "15:19:00", 523.56, 1),
    ("Bar 911", "15:38:00", 523.42, -1),
    ("Bar 926", "15:53:00", 523.56, 1),
    ("Bar 939", "16:06:00", 523.25, -1),
    ("Bar 982", "16:49:00", 523.20, 1),
    ("Bar 994", "17:01:00", 523.14, -1),
    ("Bar 995", "17:02:00", 523.22, 1),
    ("Bar 997", "17:04:00", 523.24, -1),
]

print("=== DETAILED SIGNAL COMPARISON ===\n")

print("OPTIMIZER EXTRA SIGNALS:")
print("1. 13:46:00 - BUY at $523.48 - First bar crossover (different MA init)")
print("2. 13:48:00 - SELL at $523.39 - Early reversal") 
print("3. 14:14:00 - SELL at $523.20 - Not in our validation")
print("4. 15:16:00 - BUY at $523.50 - Extra signal")
print("5. 15:18:00 - SELL at $523.50 - Quick reversal")

print("\nVALIDATION SIGNALS NOT IN OPTIMIZER:")
print("1. Bar 814 (14:01:00) - SELL at $523.49")
print("2. Bar 825 (14:12:00) - SELL at $523.05") 

print("\nANALYSIS:")
print("- Optimizer has 16 signals, we have 13")
print("- The difference is mainly in early signals and some mid-session variations")
print("- Most signals after 14:30 match well")
print("- The optimizer seems more sensitive, generating signals on smaller MA differences")