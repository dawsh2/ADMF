#!/usr/bin/env python3
import sys
sys.path.append('.')

from src.data.csv_data_handler import CSVDataHandler
from src.core.event_bus import EventBus
from src.core.config import ConfigLoader

# Create minimal setup
config_loader = ConfigLoader('config/config.yaml')
event_bus = EventBus()

# Create data handler
data_handler = CSVDataHandler(
    instance_name="debug_data_handler",
    config_loader=config_loader,
    event_bus=event_bus,
    component_config_key="components.data_handler_csv"
)

# Setup to load and split data
data_handler.setup()

print(f"\nData split analysis:")
print(f"Total data rows: {len(data_handler._data_for_run)}")
print(f"Train rows: {len(data_handler._train_df) if data_handler._train_df is not None else 0}")
print(f"Test rows: {len(data_handler._test_df) if data_handler._test_df is not None else 0}")
print(f"Split ratio: {data_handler._train_test_split_ratio}")

if data_handler._train_df is not None and not data_handler._train_df.empty:
    print(f"\nTrain data range:")
    print(f"  First: {data_handler._train_df.iloc[0]['timestamp']}")
    print(f"  Last: {data_handler._train_df.iloc[-1]['timestamp']}")

if data_handler._test_df is not None and not data_handler._test_df.empty:
    print(f"\nTest data range:")
    print(f"  First: {data_handler._test_df.iloc[0]['timestamp']}")
    print(f"  Last: {data_handler._test_df.iloc[-1]['timestamp']}")

# Test set_active_dataset
print("\n\nTesting set_active_dataset('train'):")
data_handler.set_active_dataset('train')
print(f"Active dataset size: {len(data_handler._active_df) if data_handler._active_df is not None else 0}")
if data_handler._active_df is not None and not data_handler._active_df.empty:
    print(f"Active dataset first timestamp: {data_handler._active_df.iloc[0]['timestamp']}")
    print(f"Active dataset last timestamp: {data_handler._active_df.iloc[-1]['timestamp']}")

print("\n\nTesting set_active_dataset('test'):")
data_handler.set_active_dataset('test')  
print(f"Active dataset size: {len(data_handler._active_df) if data_handler._active_df is not None else 0}")
if data_handler._active_df is not None and not data_handler._active_df.empty:
    print(f"Active dataset first timestamp: {data_handler._active_df.iloc[0]['timestamp']}")
    print(f"Active dataset last timestamp: {data_handler._active_df.iloc[-1]['timestamp']}")