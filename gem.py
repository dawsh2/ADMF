import re
import argparse
import ast # For potentially converting string dicts to dicts if clean
from collections import defaultdict
from datetime import datetime

# --- Configuration: Adjust these patterns if needed ---

# Timestamp at the beginning of the log line
# Example: "2025-05-23 20:59:44.122"
LOG_LINE_TIMESTAMP_REGEX = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})"

# For Event Payloads (extracting the whole payload string first)
EVENT_PAYLOAD_REGEX = re.compile(r"Event\(type=(\w+),.*?payload=({.*?})\)$")

# Specific extractors for fields within a payload string
# These will be applied to the captured payload string from EVENT_PAYLOAD_REGEX
SIGNAL_PAYLOAD_SYMBOL_REGEX = re.compile(r"'symbol':\s*'([^']+)'")
SIGNAL_PAYLOAD_SIGNAL_TYPE_REGEX = re.compile(r"'signal_type':\s*(-?\d+)") # Captures integer signal_type
SIGNAL_PAYLOAD_STRATEGY_ID_REGEX = re.compile(r"'strategy_id':\s*'([^']+)'")
# Optional: capture the bar timestamp from payload for context
SIGNAL_PAYLOAD_BAR_TS_REGEX = re.compile(r"'timestamp':\s*Timestamp\('([^']+)', tz='[^']+'\)")


CLASSIFICATION_PAYLOAD_CLASSIFIER_NAME_REGEX = re.compile(r"'classifier_name':\s*'([^']+)'")
CLASSIFICATION_PAYLOAD_REGIME_REGEX = re.compile(r"'classification':\s*'([^']+)'")

# Keywords to identify lines containing parameters (used if parameters are logged line-by-line AFTER a regime change)
PARAMETER_LOG_KEYWORDS = [
    "Active parameters:", "Using parameters:", "Loaded parameters for",
    "MA_SHORT:", "MA_LONG:", "MA_WEIGHT:", "short_window_default:", "long_window_default:", "ma_rule.weight:",
    "RSI_PERIOD:", "RSI_OVERSOLD:", "RSI_OVERBOUGHT:", "RSI_WEIGHT:", "rsi_rule.weight:"
]
# Regex to find a line that looks like it's logging a dictionary/JSON of parameters
PARAMETERS_JSON_REGEX = re.compile(r".* Parameters: ({.*})") # Example: "Strategy Parameters: {'key': 'value'}"
PARAM_LOOKAHEAD_LINES = 10 # Lines to check after regime change for parameters

# --- End Configuration ---

def parse_datetime_from_log_timestamp(ts_str):
    try:
        return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        print(f"Warning: Could not parse timestamp: {ts_str}")
        return None

def map_signal_type(signal_type_int_str):
    """Maps integer signal type to string (BUY, SELL, NEUTRAL)."""
    try:
        val = int(signal_type_int_str)
        if val == 1: return "BUY"
        if val == -1: return "SELL"
        if val == 0: return "NEUTRAL" # Assuming 0 is neutral
        return f"UNKNOWN_TYPE_{val}"
    except ValueError:
        return "INVALID_TYPE"

def parse_log_file(log_path, primary_regime_detector_name="MyPrimaryRegimeDetector_Instance"):
    signals = []
    regime_events = []

    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_path}")
        return [], []

    for i, line in enumerate(lines):
        line = line.strip()
        
        log_ts_match = re.search(LOG_LINE_TIMESTAMP_REGEX, line)
        if not log_ts_match:
            continue
        log_ts_str = log_ts_match.group(1)
        log_datetime = parse_datetime_from_log_timestamp(log_ts_str)

        payload_match = EVENT_PAYLOAD_REGEX.search(line)
        if not payload_match:
            continue
        
        event_type = payload_match.group(1)
        payload_str = payload_match.group(2)

        if event_type == "SIGNAL":
            symbol_match = SIGNAL_PAYLOAD_SYMBOL_REGEX.search(payload_str)
            signal_type_match = SIGNAL_PAYLOAD_SIGNAL_TYPE_REGEX.search(payload_str)
            strategy_id_match = SIGNAL_PAYLOAD_STRATEGY_ID_REGEX.search(payload_str)
            bar_ts_match = SIGNAL_PAYLOAD_BAR_TS_REGEX.search(payload_str)

            if symbol_match and signal_type_match:
                signals.append({
                    'log_timestamp_str': log_ts_str,
                    'log_datetime': log_datetime,
                    'bar_timestamp_str': bar_ts_match.group(1) if bar_ts_match else "N/A",
                    'symbol': symbol_match.group(1),
                    'signal_val': int(signal_type_match.group(1)),
                    'signal_str': map_signal_type(signal_type_match.group(1)),
                    'strategy_id': strategy_id_match.group(1) if strategy_id_match else "N/A",
                    'raw_payload': payload_str
                })

        elif event_type == "CLASSIFICATION":
            classifier_name_match = CLASSIFICATION_PAYLOAD_CLASSIFIER_NAME_REGEX.search(payload_str)
            regime_match_payload = CLASSIFICATION_PAYLOAD_REGIME_REGEX.search(payload_str)

            if classifier_name_match and classifier_name_match.group(1) == primary_regime_detector_name and regime_match_payload:
                regime_name = regime_match_payload.group(1)
                event_params = {}
                raw_param_lines_context = []

                # Look in the next few lines for parameter information explicitly logged by the strategy
                for j in range(i + 1, min(i + 1 + PARAM_LOOKAHEAD_LINES, len(lines))):
                    lookahead_line = lines[j].strip()
                    raw_param_lines_context.append(lookahead_line)

                    json_param_match = PARAMETERS_JSON_REGEX.search(lookahead_line)
                    if json_param_match:
                        try:
                            param_str_cleaned = json_param_match.group(1).replace("'", "\"").replace("None", "null").replace("True","true").replace("False","false")
                            event_params = json.loads(param_str_cleaned)
                            break 
                        except json.JSONDecodeError:
                            event_params = {"error": "JSONDecodeError", "raw_string": json_param_match.group(1)}
                            # Continue to check for keyword based if JSON fails but was found
                    
                    if not event_params or "error" in event_params: # only if not already successfully parsed as JSON
                        for keyword in PARAMETER_LOG_KEYWORDS:
                            if keyword.lower() in lookahead_line.lower(): # Case-insensitive keyword check
                                parts = re.split(f"{keyword}", lookahead_line, flags=re.IGNORECASE, maxsplit=1)
                                if len(parts) > 1:
                                    value_part = parts[1].strip()
                                    # Try to get a sensible value; this might need refinement
                                    value = value_part.split(" ")[0].split(",")[0] # First word/number
                                    param_key = keyword.replace(":", "").replace(" ", "_").lower()
                                    event_params[param_key] = value
                                # break # Found a keyword, assume one param per line for this keyword type

                regime_events.append({
                    'log_timestamp_str': log_ts_str,
                    'log_datetime': log_datetime,
                    'regime': regime_name,
                    'parameters': event_params,
                    'raw_param_lines_context': raw_param_lines_context[:5]
                })
    
    # Sort by datetime object for reliable comparison
    signals.sort(key=lambda s: s['log_datetime'] if s['log_datetime'] else datetime.max)
    regime_events.sort(key=lambda r: r['log_datetime'] if r['log_datetime'] else datetime.max)
    return signals, regime_events

def compare_signals(signals1, signals2, log1_name, log2_name):
    print("\n--- Signal Comparison ---")
    
    # Create maps for easier access: key=log_timestamp_str, value=list of signals (in case of multiple symbols)
    signals1_map = defaultdict(list)
    for s in signals1: signals1_map[s['log_timestamp_str']].append(s)
    
    signals2_map = defaultdict(list)
    for s in signals2: signals2_map[s['log_timestamp_str']].append(s)

    all_timestamps = sorted(list(set(signals1_map.keys()) | set(signals2_map.keys())))

    mismatches_count = 0
    matches_count = 0

    for ts in all_timestamps:
        s_list1 = signals1_map.get(ts, [])
        s_list2 = signals2_map.get(ts, [])

        # Simple comparison: for now, just check if signal counts match and then compare first one if counts are 1
        # A more sophisticated comparison would match signals by symbol within the same timestamp.
        
        if not s_list1 and not s_list2: # Should not happen with all_timestamps logic
            continue

        print(f"\n@ Timestamp: {ts}")
        matched_in_ts = True
        
        if len(s_list1) != len(s_list2):
            matched_in_ts = False
        else:
            # Assuming for now if counts match, we compare one-by-one (order might matter or need symbol matching)
            for i in range(len(s_list1)):
                s1 = s_list1[i]
                s2 = s_list2[i] # This assumes corresponding signals are in same order if multiple per ts
                if s1['symbol'] != s2['symbol'] or s1['signal_val'] != s2['signal_val']:
                    matched_in_ts = False
                    break
        
        log1_signals_str = ["NO SIGNAL"]
        if s_list1:
            log1_signals_str = [f"Symbol: {s['symbol']}, Signal: {s['signal_str']} (Val: {s['signal_val']}), Strategy: {s['strategy_id']}" for s in s_list1]
        
        log2_signals_str = ["NO SIGNAL"]
        if s_list2:
            log2_signals_str = [f"Symbol: {s['symbol']}, Signal: {s['signal_str']} (Val: {s['signal_val']}), Strategy: {s['strategy_id']}" for s in s_list2]

        print(f"  {log1_name}: {'; '.join(log1_signals_str)}")
        print(f"  {log2_name}: {'; '.join(log2_signals_str)}")

        if not matched_in_ts:
            print(f"  STATUS: MISMATCH")
            mismatches_count += 1
        else:
            print(f"  STATUS: MATCH")
            matches_count +=1
            
    print("\n--- Signal Comparison Summary ---")
    if not all_timestamps:
        print("No signals found in either log to compare.")
    else:
        print(f"Compared {len(all_timestamps)} unique signal timestamps.")
        print(f"Matches: {matches_count}")
        print(f"Mismatches (or signals present in only one log at a timestamp): {mismatches_count}")


def print_regime_parameters_summary(regime_events1, regime_events2, log1_name, log2_name):
    print("\n--- Regime Change Parameter Information ---")
    # As user stated regime changes are identical, we use log1 as the reference for timestamps.
    
    events2_map_by_ts_and_regime = {(e['log_timestamp_str'], e['regime']): e for e in regime_events2}

    if not regime_events1:
        print(f"No regime change events found in {log1_name} to display.")
        return

    for event1 in regime_events1:
        print(f"\nRegime Change in {log1_name} @ {event1['log_timestamp_str']} to '{event1['regime']}'")
        print(f"  Parameters ({log1_name}):")
        if event1['parameters'] and not event1['parameters'].get('error'):
            for k, v in sorted(event1['parameters'].items()):
                print(f"    {k}: {v}")
        elif event1['parameters'].get('error'):
             print(f"    Error extracting structured params: {event1['parameters']['error']}")
             print(f"    Raw string hint: {event1['parameters'].get('raw_string','')}")
             print(f"    Context lines:")
             for raw_line in event1['raw_param_lines_context']: print(f"      {raw_line}")
        else:
            print(f"    No structured parameters extracted from {log1_name}. Context lines:")
            for raw_line in event1['raw_param_lines_context']: print(f"      {raw_line}")
        
        # Attempt to find the same regime change in log2
        # This simple lookup assumes timestamps will be very close if not identical for the "same" regime change event
        # A more robust method would involve looking within a small time delta.
        event2 = None
        for ts_delta_ms in range(-50, 51, 5): # Look +/- 50ms in 5ms increments
            if not event1['log_datetime']: break
            target_dt = event1['log_datetime'] + timedelta(milliseconds=ts_delta_ms)
            target_ts_str = target_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # Format to YYYY-MM-DD HH:MM:SS.mmm
            
            potential_event2 = events2_map_by_ts_and_regime.get((target_ts_str, event1['regime']))
            if potential_event2:
                event2 = potential_event2
                break
        
        print(f"  Parameters ({log2_name}):")
        if event2:
            if event2['parameters'] and not event2['parameters'].get('error'):
                for k, v in sorted(event2['parameters'].items()):
                    print(f"    {k}: {v}")
            elif event2['parameters'].get('error'):
                print(f"    Error extracting structured params: {event2['parameters']['error']}")
                print(f"    Raw string hint: {event2['parameters'].get('raw_string','')}")
                print(f"    Context lines:")
                for raw_line in event2['raw_param_lines_context']: print(f"      {raw_line}")
            else:
                print(f"    No structured parameters extracted from {log2_name}. Context lines:")
                for raw_line in event2['raw_param_lines_context']: print(f"      {raw_line}")
        else:
            print(f"    No corresponding regime change to '{event1['regime']}' found in {log2_name} around timestamp {event1['log_timestamp_str']}.")


def main():
    parser = argparse.ArgumentParser(description="Compare signals and regime parameters from two ADMF log files based on Event Bus publishing logs.")
    parser.add_argument("log_file1", help="Path to the first log file.")
    parser.add_argument("log_file2", help="Path to the second log file.")
    parser.add_argument("--detector_name", default="MyPrimaryRegimeDetector_Instance", help="The exact name of your primary regime detector instance as logged in CLASSIFICATION events.")
    args = parser.parse_args()

    print(f"Parsing {args.log_file1}...")
    signals1, regime_events1 = parse_log_file(args.log_file1, args.detector_name)
    print(f"Found {len(signals1)} signals and {len(regime_events1)} relevant regime change events in {args.log_file1}")

    print(f"\nParsing {args.log_file2}...")
    signals2, regime_events2 = parse_log_file(args.log_file2, args.detector_name)
    print(f"Found {len(signals2)} signals and {len(regime_events2)} relevant regime change events in {args.log_file2}")

    compare_signals(signals1, signals2, args.log_file1.split('/')[-1], args.log_file2.split('/')[-1])
    print_regime_parameters_summary(regime_events1, regime_events2, args.log_file1.split('/')[-1], args.log_file2.split('/')[-1])

if __name__ == "__main__":
    from datetime import timedelta # Place here to avoid import error if script not run as main
    main()
