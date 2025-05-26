# Logging Optimization Guide

This guide outlines strategies for optimizing logging output to improve analysis and reduce verbosity in the ADMF system.

## Core Issues

1. **Excessive Debug Output**: Particularly from regime detection and classification
2. **Log Level Misuse**: Using INFO for detailed debugging messages
3. **No Filtering Mechanism**: Inability to toggle verbosity per component
4. **Analysis Challenges**: Difficulty extracting insights from large log files

## Implementation Plan

### 1. Log Level Adjustment

**Target Files:**
- `src/strategy/regime_detector.py`
- `src/core/logging_setup.py`

**Changes:**
```python
# Before
self.logger.info("OPTIMIZATION DEBUG - Checking against thresholds: ...")

# After
self.logger.debug("Checking against thresholds: ...")
```

**Configuration:**
```yaml
logging:
  level: "INFO"  # System default
  component_levels:
    MyPrimaryRegimeDetector: "INFO"  # Specific override
```

### 2. Conditional Logging

**Target Files:**
- `src/strategy/regime_detector.py`

**Changes:**
```python
# Add configuration field
self.verbose_logging = self.get_config("verbose_logging", False)

# Conditional logging
if self.verbose_logging:
    self.logger.debug("Detailed regime check: ...")
```

**Configuration:**
```yaml
components:
  MyPrimaryRegimeDetector:
    verbose_logging: false  # Only enable when debugging
```

### 3. Event-Based Logging

**Implementation:**
- Only log full details when regime actually changes
- Track statistics and emit summary logs at intervals

```python
# Track stats instead of logging each check
self._checks_since_last_log += 1
self._no_match_count += 1

# Emit summary periodically
if self._checks_since_last_log >= 100:
    self.logger.info(f"Regime detection summary: {self._no_match_count}/100 no matches")
    self._checks_since_last_log = 0
    self._no_match_count = 0
```

### 4. Summary Reporting

**Implementation:**
- Add end-of-run summary generation
- Create specific regime detection metrics

```python
def generate_summary(self):
    """Generate regime detection summary at end of run."""
    self.logger.info("=== Regime Detection Summary ===")
    self.logger.info(f"Total checks: {self._total_checks}")
    self.logger.info(f"No match rate: {(self._no_match_count / self._total_checks) * 100:.2f}%")
    for regime, count in self._regime_counts.items():
        self.logger.info(f"Regime '{regime}': {count} instances ({(count / self._total_checks) * 100:.2f}%)")
```

## Implementation Priority

1. **Log Level Adjustment**: Immediate impact, minimal changes
2. **Conditional Logging**: Add configuration toggle for verbose output
3. **Summary Statistics**: Add counters and periodic summaries
4. **End-of-Run Reporting**: Generate comprehensive report at completion

## Configuration Example

```yaml
logging:
  level: "INFO"  # System default
  component_levels:
    MyPrimaryRegimeDetector: "INFO"

components:
  MyPrimaryRegimeDetector:
    # Existing configuration...
    min_regime_duration: 2
    # Logging control
    verbose_logging: false
    summary_interval: 100  # Log summary every N checks
```

## Testing Approach

1. Run with default settings to establish baseline log size
2. Apply log level changes and compare output size
3. Add conditional logging and verify toggle functionality
4. Implement summary reporting and verify insights quality

## Expected Outcomes

- **Log Size Reduction**: 70-90% smaller log files
- **Improved Analysis**: Focus on meaningful events and summary statistics
- **Configurable Verbosity**: Toggle detailed logs only when needed
- **Better Insights**: Summary reports highlight key metrics