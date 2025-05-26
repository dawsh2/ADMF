#!/usr/bin/env python3
"""
This script patches the EnhancedOptimizer to work with the refactored BasicOptimizer.
It updates references from self.name to self.instance_name and self._container to self.container.
"""
import re

def patch_enhanced_optimizer():
    # Read the original file
    with open('/Users/daws/ADMF/src/strategy/optimization/enhanced_optimizer.py', 'r') as f:
        content = f.read()
    
    # Apply replacements
    replacements = [
        # Update references to self.name -> self.instance_name
        (r'\bself\.name\b', 'self.instance_name'),
        # Update references to self._container -> self.container
        (r'\bself\._container\b', 'self.container'),
        # Update references to self._event_bus -> self.event_bus
        (r'\bself\._event_bus\b', 'self.event_bus'),
        # Update STATE_ constants
        (r'BasicOptimizer\.STATE_STARTED', 'self.ComponentState.RUNNING'),
        (r'BasicOptimizer\.STATE_STOPPED', 'self.ComponentState.STOPPED'),
        (r'BasicOptimizer\.STATE_FAILED', 'self.ComponentState.FAILED'),
        (r'BasicOptimizer\.STATE_INITIALIZED', 'self.ComponentState.INITIALIZED'),
        # Update get_state() calls
        (r'\.get_state\(\)', '.state'),
        # Update instance references in resolve calls
        (r'regime_detector\.name', 'regime_detector.instance_name'),
        (r'comp\.name', 'comp.instance_name'),
        (r'genetic_optimizer\.name', 'genetic_optimizer.instance_name'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Update the constructor to call parent with minimal parameters
    constructor_pattern = r'def __init__\(self, instance_name: str, config_loader, event_bus, component_config_key: str, container\):\s*super\(\).__init__\(instance_name, config_loader, event_bus, component_config_key, container\)'
    constructor_replacement = '''def __init__(self, instance_name: str, config_key: Optional[str] = None):
        """Minimal constructor following ComponentBase pattern."""
        super().__init__(instance_name, config_key)'''
    
    content = re.sub(constructor_pattern, constructor_replacement, content, flags=re.DOTALL)
    
    # Move initialization logic from constructor to _initialize method
    # Find the initialization code in constructor
    init_pattern = r'(# Settings specific to enhanced optimizer.*?)(?=self\.logger\.info)'
    match = re.search(init_pattern, content, re.DOTALL)
    
    if match:
        init_code = match.group(1)
        
        # Remove this code from constructor
        content = content.replace(init_code, '')
        
        # Add _initialize method after constructor
        initialize_method = f'''
    
    def _initialize(self):
        """Component-specific initialization logic."""
        # Call parent's _initialize first
        super()._initialize()
        
        {init_code}'''
        
        # Insert after the constructor
        content = re.sub(r'(super\(\).__init__.*?\n)', r'\1' + initialize_method + '\n', content)
    
    # Write the patched file
    with open('/Users/daws/ADMF/src/strategy/optimization/enhanced_optimizer.py', 'w') as f:
        f.write(content)
    
    print("Enhanced optimizer patched successfully!")

if __name__ == '__main__':
    patch_enhanced_optimizer()