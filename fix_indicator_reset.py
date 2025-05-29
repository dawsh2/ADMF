#!/usr/bin/env python3
"""
Script to fix indicator reset methods to ensure true cold start.
"""

import os
import re

def fix_indicator_reset(file_path, indicator_type):
    """Fix reset method in indicator file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the reset method
    reset_pattern = r'(def reset\(self\) -> None:\s*\n\s*"""[^"]*"""\s*\n)(.*?)(\n\s*def|\n\s*@|\Z)'
    
    def replacement(match):
        method_def = match.group(1)
        old_body = match.group(2)
        next_method = match.group(3)
        
        # Create new reset body based on indicator type
        if indicator_type == 'base':
            new_body = '''        if hasattr(self, 'logger') and self.logger:
            self.logger.info(f"Indicator {self.instance_name} RESET - clearing {len(self._buffer)} bars of history")
        
        # Reinitialize buffer to ensure true cold start
        self._buffer = []  # Start with empty list
        self._value = None
        self._ready = False
        '''
        elif indicator_type == 'bollinger':
            new_body = '''        self.logger.info(f"BollingerBands {self.instance_name} RESET - clearing buffer")
        
        # Reinitialize buffer to ensure true cold start
        self._price_buffer = deque(maxlen=self.lookback_period)
        self._upper_band = None
        self._lower_band = None
        self._middle_band = None
        '''
        elif indicator_type == 'macd':
            new_body = '''        self.logger.info(f"MACD {self.instance_name} RESET - clearing buffers")
        
        # Reinitialize all buffers to ensure true cold start
        self._price_buffer = deque(maxlen=max(self.slow_period, self.fast_period, self.signal_period))
        self._fast_ema = None
        self._slow_ema = None
        self._signal_ema = None
        self._macd_line = None
        self._signal_line = None
        self._histogram = None
        '''
        elif indicator_type == 'regime_indicators':
            new_body = '''        self.logger.info(f"Indicator {self.instance_name} RESET - clearing {len(self._buffer)} bars of history")
        
        # Reinitialize buffer to ensure true cold start
        self._buffer = deque(maxlen=self._lookback_period)
        self._value = None
        self._ready = False
        '''
        else:
            return match.group(0)  # Return unchanged
        
        return method_def + new_body + next_method
    
    new_content = re.sub(reset_pattern, replacement, content, flags=re.DOTALL)
    
    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"✓ Fixed reset method in {file_path}")
        return True
    else:
        print(f"✗ No changes needed in {file_path}")
        return False

# Fix base indicator class
print("Fixing indicator reset methods...")
print("="*60)

# Fix base indicator
fix_indicator_reset('src/strategy/base/indicator.py', 'base')

# Fix Bollinger Bands
fix_indicator_reset('src/strategy/components/indicators/bollinger_bands.py', 'bollinger')

# Fix MACD
fix_indicator_reset('src/strategy/components/indicators/macd.py', 'macd')

# Fix regime detector indicators
for ind_file in ['trend.py', 'oscillators.py']:
    fix_indicator_reset(f'src/strategy/components/indicators/{ind_file}', 'regime_indicators')

print("\n" + "="*60)
print("Indicator reset fixes complete!")