# Component Refactoring Plan

## Overview

This document outlines the plan for refactoring existing components to use the new ComponentBase interface.

## Current Component Status

### Components Needing Refactoring:

1. **Data Components**
   - `CSVDataHandler` → Needs ComponentBase inheritance
   
2. **Risk Components**
   - `BasicPortfolio` → Needs ComponentBase inheritance
   - `BasicRiskManager` → Needs ComponentBase inheritance
   
3. **Execution Components**
   - `SimulatedExecutionHandler` → Needs ComponentBase inheritance
   
4. **Strategy Components**
   - `MAStrategy` → Needs ComponentBase inheritance
   - `RegimeAdaptiveStrategy` → Needs ComponentBase inheritance
   - `RegimeDetector` → Needs ComponentBase inheritance
   
5. **Optimization Components**
   - `BasicOptimizer` → Needs ComponentBase inheritance
   - `EnhancedOptimizer` → Needs ComponentBase inheritance
   - `GeneticOptimizer` → Needs ComponentBase inheritance

## Refactoring Pattern

Each component needs to be updated following this pattern:

### Before (Current):
```python
class CSVDataHandler(BaseComponent):
    def __init__(self, instance_name, config_loader, event_bus, component_config_key, max_bars=None):
        super().__init__(instance_name, config_loader, event_bus, component_config_key)
        # Complex initialization with external dependencies
```

### After (Target):
```python
class CSVDataHandler(ComponentBase):
    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        super().__init__(instance_name, config_key)
        # Minimal - no external dependencies
        
    def _initialize(self) -> None:
        # Move dependency acquisition here
        # Access config via self.component_config
        # Set up internal state
        
    def initialize_event_subscriptions(self) -> None:
        # Use self.subscription_manager
        self.subscription_manager.subscribe(EventType.SYSTEM, self.on_system_event)
```

## Priority Order

### Phase 1: Core Components (CRITICAL) ✅
1. **CSVDataHandler** - Data flow foundation ✅
2. **BasicPortfolio** - State management critical ✅
3. **MAStrategy** - Basic strategy for testing ✅

### Phase 2: Risk & Execution ✅
4. **BasicRiskManager** ✅
5. **SimulatedExecutionHandler** ✅

### Phase 3: Advanced Strategies ✅
6. **RegimeAdaptiveStrategy** ✅
7. **RegimeDetector** ✅

### Phase 4: Optimizers (IN PROGRESS)
8. **BasicOptimizer** ✅
9. **EnhancedOptimizer** ✅ (Patched to work with refactored BasicOptimizer)
10. **GeneticOptimizer**
11. **OptimizationRunner** ✅ (New - Implements modular optimization framework)

## Key Refactoring Points

### 1. Constructor Simplification
- Remove all external dependencies from `__init__`
- Only accept `instance_name` and `config_key`

### 2. Initialization Phase
- Move dependency acquisition to `_initialize()`
- Use `self.context` to access:
  - `self.event_bus`
  - `self.container`
  - `self.config`
  - `self.logger`

### 3. Event Subscriptions
- Override `initialize_event_subscriptions()`
- Use `self.subscription_manager.subscribe()`
- No manual unsubscription needed (handled by teardown)

### 4. State Reset
- Implement proper `reset()` method
- Clear all state except configuration
- Prepare for fresh run

### 5. Resource Cleanup
- Override `teardown()` if needed
- Release external resources
- Call `super().teardown()`

## Testing Strategy

1. **Unit Tests** - Test each component in isolation
2. **Integration Tests** - Test component interactions
3. **Backward Compatibility** - Ensure legacy usage still works
4. **Scoped Container Tests** - Verify isolation

## Migration Approach

### Step 1: Create Adapter
Create temporary adapter to make old components work with new system

### Step 2: Refactor Incrementally
Refactor one component at a time, testing thoroughly

### Step 3: Update Bootstrap
Update STANDARD_COMPONENTS as each component is refactored

### Step 4: Remove Legacy Code
Once all components migrated, remove old BaseComponent

## Success Criteria

- [ ] All components inherit from ComponentBase
- [ ] No external dependencies in constructors
- [ ] All use SubscriptionManager for events
- [ ] Clean state reset capability
- [ ] Pass all existing tests
- [ ] Work with scoped containers

## Timeline Estimate

- Phase 1: 1-2 days
- Phase 2: 1 day
- Phase 3: 1 day
- Phase 4: 2 days
- Testing & Integration: 2 days

Total: ~1 week for full migration