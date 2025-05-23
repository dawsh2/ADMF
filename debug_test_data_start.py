#!/usr/bin/env python3
"""Debug why production test data starts at wrong date"""

import pandas as pd

# Load the data
df = pd.read_csv('data/SPY_1min.csv')
print(f"Total rows: {len(df)}")

# Check train/test split point
split_point = int(len(df) * 0.8)
print(f"\n80/20 split point: index {split_point}")

# Show what's at the split point
print(f"\nData at split point (index {split_point}):")
print(df.iloc[split_point-2:split_point+3][['timestamp', 'Open', 'Close']])

# Find January 2 and January 27
jan2_mask = df['timestamp'].str.contains('2025-01-02')
jan27_mask = df['timestamp'].str.contains('2025-01-27')

jan2_indices = df[jan2_mask].index
jan27_indices = df[jan27_mask].index

if len(jan2_indices) > 0:
    print(f"\nJanuary 2, 2025 starts at index: {jan2_indices[0]}")
    print(f"Distance from split point: {jan2_indices[0] - split_point}")
    
if len(jan27_indices) > 0:
    print(f"\nJanuary 27, 2025 starts at index: {jan27_indices[0]}")
    print(f"Distance from split point: {jan27_indices[0] - split_point}")

# Check if there's any filtering or sorting happening
print(f"\nFirst few test indices should be:")
test_df = df.iloc[split_point:split_point+5]
print(test_df[['timestamp', 'Open', 'Close']])