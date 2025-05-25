#!/usr/bin/env python3
"""
Search for adaptive mode messages in log files
"""
import os
import re

def search_log_file(log_path, output_file):
    """Search for adaptive/regime related messages in log file"""
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return
    
    print(f"Searching {log_path}...")
    
    # Patterns to search for
    patterns = [
        r'.*[Aa][Dd][Aa][Pp][Tt][Ii][Vv][Ee].*',
        r'.*[Rr][Ee][Gg][Ii][Mm][Ee].*',
        r'.*parameters.*',
        r'.*MODE.*',
        r'.*adaptive.*',
        r'.*ADAPTIVE.*'
    ]
    
    matches = []
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        matches.append(f"Line {line_num}: {line.strip()}")
                        break  # Avoid duplicate matches for same line
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return
    
    # Write results to output file
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"\n=== Results from {log_path} ===\n")
        if matches:
            f.write(f"Found {len(matches)} matches:\n\n")
            for match in matches:
                f.write(f"{match}\n")
        else:
            f.write("No adaptive/regime related messages found.\n")
        f.write("\n" + "="*50 + "\n")
    
    print(f"Found {len(matches)} matches in {log_path}")

if __name__ == "__main__":
    # Output file for results
    output_file = "/Users/daws/ADMF/adaptive_search_results.txt"
    
    # Clear output file
    with open(output_file, 'w') as f:
        f.write("Adaptive Mode Log Search Results\n")
        f.write("="*50 + "\n")
    
    # Search both log files
    log_files = [
        "/Users/daws/ADMF/logs/admf_20250523_212214.log",
        "/Users/daws/ADMF/logs/admf_20250523_211602.log"
    ]
    
    for log_file in log_files:
        search_log_file(log_file, output_file)
    
    print(f"\nResults saved to: {output_file}")