#!/usr/bin/env python3
"""Debug why production is using training data instead of test data"""

import pandas as pd
import sys
sys.path.append('/Users/daws/ADMF')

# Load the CSV file
df = pd.read_csv('data/SPY_1min.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# With --bars 10000
limited_df = df.head(10000)

# Apply 80/20 split
split_point = int(len(limited_df) * 0.8)
train_df = limited_df.iloc[:split_point]
test_df = limited_df.iloc[split_point:]

print("DATASET ANALYSIS WITH --bars 10000")
print("="*60)
print(f"Total bars: {len(limited_df)}")
print(f"Train bars: {len(train_df)} (bars 0-{split_point-1})")
print(f"Test bars: {len(test_df)} (bars {split_point}-{len(limited_df)-1})")

print("\nTRAIN DATA:")
print(f"  First timestamp: {train_df['timestamp'].iloc[0]}")
print(f"  Last timestamp: {train_df['timestamp'].iloc[-1]}")
print(f"  Date range: {train_df['timestamp'].iloc[0].date()} to {train_df['timestamp'].iloc[-1].date()}")

print("\nTEST DATA:")
print(f"  First timestamp: {test_df['timestamp'].iloc[0]}")
print(f"  Last timestamp: {test_df['timestamp'].iloc[-1]}")
print(f"  Date range: {test_df['timestamp'].iloc[0].date()} to {test_df['timestamp'].iloc[-1].date()}")

print("\nPRODUCTION LOG SHOWS:")
print("  First timestamp: 2024-04-24 16:30:00 (this is TRAINING data!)")

print("\nCONCLUSION:")
print("  Production is incorrectly using TRAINING data even though it says 'Setting active dataset to test'")
print("  This is why results don't match - it's running on completely different data!")