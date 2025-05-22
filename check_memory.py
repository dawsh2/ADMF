#!/usr/bin/env python3
import psutil
import os

def check_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    
    print(f"Memory usage: {memory_mb:.1f} MB")
    
    # System memory
    system_memory = psutil.virtual_memory()
    print(f"System memory: {system_memory.percent}% used")
    print(f"Available: {system_memory.available / 1024 / 1024 / 1024:.1f} GB")

if __name__ == "__main__":
    check_memory()