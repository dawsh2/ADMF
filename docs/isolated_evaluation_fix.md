# Isolated Evaluation Container Fix

## Changes Made

### 1. Complete Container Isolation
- Created `_create_isolated_container()` method that creates a fresh container with:
  - New EventBus instance
  - New Context instance
  - Fresh Portfolio (no P&L contamination)
  - Fresh Risk Manager
  - Fresh Execution Handler
  - Fresh Data Handler (with copied data)
  - Fresh Backtest Runner

### 2. Component Isolation
- Created `_create_component_copy()` method that:
  - Creates fresh instance of the component being tested
  - Creates fresh instances of all indicator dependencies
  - Re-initializes everything with the isolated context
  - Applies current parameters to the fresh component

### 3. Proper Cleanup
- Stop all components in reverse order
- Teardown all components
- Reset the isolated container
- Log portfolio state before and after for verification

## Key Benefits
1. **No State Leakage**: Each evaluation runs in complete isolation
2. **No P&L Contamination**: Fresh portfolio starts at initial cash
3. **Fresh Event Bus**: No lingering event subscriptions
4. **Independent Results**: Each component evaluation is truly independent

## Testing
To verify the fix works:
1. Check logs for "Isolated portfolio initial state" - should show:
   - Cash=100000.0
   - Realized P&L=0.0
   - Total Value=100000.0

2. Verify each component gets different scores based on actual performance
3. Confirm no "drift returns" - returns should be 0 if no trades occur