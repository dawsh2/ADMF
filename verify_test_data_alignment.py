#!/usr/bin/env python3
"""
Verify that OOS test and production use the same test dataset.
"""

import pandas as pd
from src.core.config import SimpleConfigLoader

# Load config
config = SimpleConfigLoader("config/config.yaml")

# Load data
csv_path = config.get("components.data_handler_csv.csv_file_path", "data/1000_1min.csv")
df = pd.read_csv(csv_path)

# Calculate split
split_ratio = config.get("components.data_handler_csv.train_test_split_ratio", 0.8)
split_index = int(len(df) * split_ratio)

print(f"Total rows: {len(df)}")
print(f"Split ratio: {split_ratio}")
print(f"Split index: {split_index}")
print(f"Train rows: {split_index}")
print(f"Test rows: {len(df) - split_index}")
print()
print("Test data starts at:")
print(f"  Index: {split_index}")
print(f"  Timestamp: {df.iloc[split_index]['timestamp']}")
print()
print("Test data ends at:")
print(f"  Index: {len(df)-1}")
print(f"  Timestamp: {df.iloc[-1]['timestamp']}")
print()
print("First 5 test rows:")
print(df.iloc[split_index:split_index+5][['timestamp', 'Close']])