# ADMF System Handover Checklist

## System Status Overview

### ✅ What's Working
- **Backtest Mode** - Fully functional with new ComponentBase architecture
- **File Logging** - Automatic log files created in `logs/` directory
- **CLI Arguments** - Properly passed to components (e.g., `--bars 50`)
- **Regime Detection** - Working with indicators and statistics
- **Component Lifecycle** - Clean initialization, startup, and teardown
- **Event System** - Proper event flow between components
- **Total Return Calculation** - Shows actual percentage returns

### ⚠️ What Needs Attention
- **Optimization Mode** - Not implemented in new architecture (use `main.py --optimize`)
- **P&L Discrepancy** - Minor ~$5 difference between displayed and calculated values (see FIX_ME.MD)

## Quick Start Commands

### Running a Backtest
```bash
# Basic backtest
python main_ultimate.py --config config/config.yaml --bars 50

# Full dataset backtest
python main_ultimate.py --config config/config.yaml

# With debug logging to console
python main_ultimate.py --config config/config.yaml --log-level DEBUG
```

### Running Optimization
```bash
# Use the original main.py for optimization
python main.py --config config/config.yaml --optimize --bars 100

# Optimize specific parameters
python main.py --config config/config.yaml --optimize-ma --bars 100
python main.py --config config/config.yaml --optimize-rsi --bars 100
```

## Key Architecture Changes

### New Component Pattern
All components now inherit from `ComponentBase` and follow this lifecycle:
1. **Constructor** - Minimal, only takes `instance_name` and `config_key`
2. **initialize(context)** - Receives dependencies via context dictionary
3. **_initialize()** - Component-specific initialization
4. **_start()** - Component startup logic
5. **stop()** - Component shutdown
6. **teardown()** - Resource cleanup

### Example Component Structure
```python
class MyComponent(ComponentBase):
    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        super().__init__(instance_name, config_key)
        # Only internal state initialization
        
    def _initialize(self):
        """Component-specific initialization."""
        # Load config, setup internal structures
        
    def _start(self):
        """Start the component."""
        # Begin processing
```

## Important Files

### Core System
- `main_ultimate.py` - New main entry point
- `src/core/bootstrap.py` - Component lifecycle management
- `src/core/component_base.py` - Base class for all components
- `src/core/application_launcher.py` - Application startup logic

### Configuration
- `config/config.yaml` - Main configuration file
- `config/config_optimization.yaml` - Optimization configuration (incomplete)

### Documentation
- `FIX_ME.MD` - Known issues and bugs
- `OPTIMIZATION_USAGE.md` - How to run optimization
- `logs/` - All system logs with timestamps

## Key Infrastructure Features

### Already Implemented
1. **Scoped Contexts** - `bootstrap.create_scoped_context()` for isolation
2. **Dependency Injection** - Via context dictionary in `initialize()`
3. **Event Bus** - Isolated per scope
4. **Component Reset** - Portfolio and strategy reset methods
5. **Parameter Versioning** - Full framework documented

### Not Yet Integrated
1. **Optimization Runner** - Needs to use scoped contexts properly
2. **Parameter Injection** - Optimization framework exists but not connected
3. **Result Aggregation** - Framework exists but not used

## Common Issues & Solutions

### Issue: "No active dataset selected"
**Solution**: Ensure data handler's `set_active_dataset()` is called before streaming

### Issue: "PortfolioManager not linked"
**Solution**: Check component initialization order - portfolio must init before risk manager

### Issue: Regime detector shows "No indicators configured"
**Solution**: Check config file has proper regime_indicators configuration

### Issue: No trades generated
**Solution**: Check if risk manager is properly initialized and linked to portfolio

## Development Tips

1. **Always use INFO logging by default** - DEBUG generates very large files
2. **Check logs directory** - Detailed debug logs are always saved there
3. **Use --bars flag** - Limits data for faster testing
4. **Component dependencies** - Declared in STANDARD_COMPONENTS in bootstrap.py
5. **Reset between tests** - Components maintain state, use fresh instances

## Next Steps for Optimization

The optimization infrastructure exists but needs integration:

1. **Use Scoped Contexts** - Each optimization iteration needs isolation
2. **Leverage Existing Framework** - See `docs/modules/strategy/optimization/`
3. **Connect Parameter System** - Use the versioned parameter framework
4. **Implement Result Storage** - Use the existing results management

The building blocks are all there in the documentation - they just need to be connected properly in a new OptimizationEntrypoint component.

## Contact & Support

For issues or questions:
1. Check `FIX_ME.MD` for known issues
2. Review logs in `logs/` directory
3. Consult documentation in `docs/` directory
4. Component-specific docs in `docs/modules/`