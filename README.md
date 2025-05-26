# ADMF-Trader: Adaptive Decision Making Framework

A sophisticated algorithmic trading framework built on a component-based architecture with advanced regime detection and optimization capabilities.

## Overview

ADMF-Trader is a production-grade backtesting and trading system designed for quantitative trading strategy development. It features a robust event-driven architecture, comprehensive risk management, and adaptive strategies that adjust to changing market conditions.

### Key Features

- **Component-Based Architecture**: Clean separation of concerns with dependency injection
- **Event-Driven Design**: Asynchronous message passing between components
- **Regime Detection**: Adaptive strategies that respond to market conditions
- **Advanced Optimization**: Multiple optimization algorithms including genetic optimization
- **State Isolation**: Reproducible results through proper component lifecycle management
- **Memory Efficient**: Multiple data isolation strategies for handling large datasets

## Architecture

The system is built on a solid foundation of core components:

```
src/
├── core/           # Foundation: ComponentBase, EventBus, Container, Bootstrap
├── data/           # Market data handling and train/test splitting
├── execution/      # Order execution and backtest simulation
├── risk/           # Portfolio management and risk controls
└── strategy/       # Trading strategies and optimization
```

### Core Components

- **ComponentBase**: Base class providing lifecycle management for all components
- **EventBus**: Publish-subscribe system for loose coupling between components
- **Container**: Dependency injection for clean component creation
- **Bootstrap**: Application initialization and component orchestration

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/admf-trader.git
cd admf-trader

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running a Backtest

```bash
# Run a simple backtest
python main.py --config config/config.yaml

# Run with limited bars for testing
python main.py --config config/config.yaml --bars 100

# Run optimization
python main.py --config config/optimization_config.yaml --mode optimize
```

### Configuration

The system uses YAML configuration files. Example configuration:

```yaml
run_mode: backtest

components:
  backtest_runner:
    class_path: src.execution.backtest_runner.BacktestRunner
    config:
      use_test_dataset: false
      close_positions_at_end: true
      
  data_handler:
    class_path: src.data.csv_data_handler.CSVDataHandler
    config:
      csv_file_path: data/SPY_1min.csv
      symbol: SPY
      train_test_split:
        train_ratio: 0.8
        
  strategy:
    class_path: src.strategy.implementations.ensemble_strategy.EnsembleStrategy
    config:
      symbol: SPY
      regime_adaptation:
        enabled: true
```

## Component Overview

### Data Module
Handles market data loading, preprocessing, and train/test splitting with proper isolation to prevent data leakage.

### Execution Module
Manages order lifecycle, simulates market fills with realistic slippage and commission models.

### Risk Module
- **Portfolio Management**: Tracks positions, P&L, and performance metrics
- **Risk Manager**: Converts signals to orders with position sizing
- **Performance Analytics**: Comprehensive metrics including regime-specific performance

### Strategy Module
- **Base Strategies**: Moving average crossover, RSI-based rules
- **Regime Adaptation**: Strategies that adapt parameters based on market conditions
- **Ensemble Strategies**: Combine multiple signals with configurable weights

### Optimization Module
- **Grid Search**: Exhaustive parameter search
- **Genetic Optimization**: Evolutionary algorithms for complex parameter spaces
- **Regime-Specific**: Optimize parameters for different market conditions

## Data Format

The system expects CSV files with the following columns:
- `timestamp`: DateTime index
- `open`, `high`, `low`, `close`: OHLC price data
- `volume`: Trading volume
- Additional columns are preserved and accessible to strategies

## Development Status

### Current State
- ✅ Core architecture complete and stable
- ✅ Basic backtesting functional
- ✅ Event-driven system working
- ✅ Trade execution pipeline complete
- 🚧 Strategy module refactoring in progress
- 🚧 Advanced optimization methods being added

### Roadmap
See [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) for detailed development plans.

## Testing

```bash
# Run unit tests
pytest tests/

# Run specific test file
pytest tests/test_core_components.py

# Run with coverage
pytest --cov=src tests/
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:
- `docs/modules/` - Detailed module documentation
- `docs/modules/core/` - Core architecture and design patterns
- `docs/modules/strategy/` - Strategy development guide
- `docs/modules/risk/` - Risk management documentation

## Performance Considerations

- The system is designed to handle datasets with millions of bars
- Memory-efficient data structures for large-scale backtesting
- Event bus optimized for high-frequency updates
- Parallel optimization support for compute-intensive searches

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Python 3.9+
- Uses pandas for data manipulation
- NumPy for numerical computations
- YAML for configuration management

## Support

For questions or issues:
- Open an issue on GitHub
- Check existing documentation in `docs/`
- Review test cases for usage examples

---

**Note**: This is an active research project. While the core architecture is stable, APIs may change as we refine the system.