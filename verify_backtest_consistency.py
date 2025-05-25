#!/usr/bin/env python3
"""
Verify Backtest Consistency - Demonstrates that BacktestEngine produces
identical results when used in optimizer vs production contexts.
"""

import json
import logging
import sys
from pathlib import Path

from src.core.config import SimpleConfigLoader
from src.core.logging_setup import setup_logging
from src.core.event_bus import EventBus
from src.core.container import Container
from src.strategy.optimization.engines import BacktestEngine
from src.strategy.optimization.enhanced_optimizer_v2 import EnhancedOptimizerV2

# Import components
from src.data.csv_data_handler import CSVDataHandler
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy
from src.strategy.regime_detector import RegimeDetector
from src.risk.basic_risk_manager import BasicRiskManager
from src.execution.simulated_execution_handler import SimulatedExecutionHandler
from src.risk.basic_portfolio import BasicPortfolio
from src.strategy.regime_adaptive_strategy import RegimeAdaptiveStrategy

logger = logging.getLogger(__name__)


def setup_container(config_loader, event_bus):
    """Set up container with all required components."""
    container = Container()
    container.register_instance("config_loader", config_loader)
    container.register_instance("event_bus", event_bus)
    container.register_instance("container", container)
    
    # Register all required components
    # Data handler
    csv_args = {
        "instance_name": "SPY_CSV_Loader",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.data_handler_csv"
    }
    container.register_type("data_handler", CSVDataHandler, True, constructor_kwargs=csv_args)
    
    # Strategies
    ensemble_args = {
        "instance_name": "SPY_Ensemble_Strategy",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.ensemble_strategy",
        "container": container
    }
    container.register_type("strategy", EnsembleSignalStrategy, True, constructor_kwargs=ensemble_args)
    container.register_type("ensemble_strategy", EnsembleSignalStrategy, True, constructor_kwargs=ensemble_args)
    
    adaptive_args = {
        "instance_name": "RegimeAdaptiveStrategy",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "container": container,
        "component_config_key": "components.regime_adaptive_strategy"
    }
    container.register_type("regime_adaptive_strategy", RegimeAdaptiveStrategy, True, constructor_kwargs=adaptive_args)
    
    # Regime detector
    regime_args = {
        "instance_name": "MyPrimaryRegimeDetector_Instance",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.MyPrimaryRegimeDetector"
    }
    container.register_type("MyPrimaryRegimeDetector", RegimeDetector, True, constructor_kwargs=regime_args)
    
    # Portfolio
    portfolio_args = {
        "instance_name": "BasicPortfolio",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "container": container,
        "component_config_key": "components.basic_portfolio"
    }
    container.register_type("portfolio_manager", BasicPortfolio, True, constructor_kwargs=portfolio_args)
    
    # Risk manager
    risk_args = {
        "instance_name": "BasicRiskMan1",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.basic_risk_manager",
        "container": container,
        "portfolio_manager_key": "portfolio_manager"
    }
    container.register_type("risk_manager", BasicRiskManager, True, constructor_kwargs=risk_args)
    
    # Execution handler
    exec_args = {
        "instance_name": "SimExec_1",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.simulated_execution_handler"
    }
    container.register_type("execution_handler", SimulatedExecutionHandler, True, constructor_kwargs=exec_args)
    
    # Optimizer
    optimizer_args = {
        "instance_name": "EnhancedOptimizer",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.optimizer",
        "container": container
    }
    container.register_type("optimizer_service", EnhancedOptimizerV2, True, constructor_kwargs=optimizer_args)
    
    return container


def run_direct_backtest(backtest_engine, params_file):
    """Run a backtest directly using BacktestEngine."""
    logger.info("Running direct backtest with BacktestEngine")
    
    metric_value, regime_performance = backtest_engine.run_backtest(
        parameters={},
        dataset_type="test",
        strategy_type="regime_adaptive",
        use_regime_adaptive=True,
        adaptive_params_path=params_file
    )
    
    return metric_value, regime_performance


def run_optimizer_adaptive_test(container, params_file):
    """Run adaptive test through optimizer."""
    logger.info("Running adaptive test through optimizer")
    
    # Load the parameters to create a mock results structure
    with open(params_file, 'r') as f:
        regime_params = json.load(f)
        
    # Create results structure that optimizer expects
    results = {
        "best_parameters_per_regime": {},
        "best_metric_per_regime": {}
    }
    
    # Format parameters for optimizer
    for regime, params in regime_params.items():
        results["best_parameters_per_regime"][regime] = {
            "parameters": params,
            "metric": {"name": "sharpe_ratio", "value": 0.0}
        }
        
    # Get optimizer and run adaptive test
    optimizer = container.resolve("optimizer_service")
    optimizer.setup()
    
    # Set up train/test split for optimizer
    data_handler = container.resolve("data_handler")
    data_handler.setup()
    data_handler.start()
    
    # Create mock train/test DataFrames for optimizer
    if hasattr(data_handler, 'data_frame'):
        split_index = int(len(data_handler.data_frame) * 0.8)
        optimizer.train_df = data_handler.data_frame.iloc[:split_index]
        optimizer.test_df = data_handler.data_frame.iloc[split_index:]
    
    # Run adaptive test
    updated_results = optimizer.run_adaptive_test(results)
    
    if "regime_adaptive_test_results" in updated_results:
        adaptive_results = updated_results["regime_adaptive_test_results"]
        metric_value = adaptive_results.get("adaptive_metric")
        regime_performance = adaptive_results.get("adaptive_regime_performance")
        return metric_value, regime_performance
    else:
        return None, None


def compare_results(direct_metric, direct_perf, optimizer_metric, optimizer_perf):
    """Compare results from direct and optimizer runs."""
    print("\n" + "=" * 80)
    print("BACKTEST CONSISTENCY VERIFICATION")
    print("=" * 80)
    
    # Compare final metrics
    print(f"\nFinal Portfolio Values:")
    print(f"  Direct BacktestEngine:    {direct_metric:.2f}" if direct_metric else "  Direct: Failed")
    print(f"  Optimizer Adaptive Test:  {optimizer_metric:.2f}" if optimizer_metric else "  Optimizer: Failed")
    
    if direct_metric and optimizer_metric:
        difference = abs(direct_metric - optimizer_metric)
        pct_diff = (difference / direct_metric) * 100
        print(f"  Difference: {difference:.2f} ({pct_diff:.4f}%)")
        
        if pct_diff < 0.01:  # Less than 0.01% difference
            print("\n✅ RESULTS ARE CONSISTENT!")
        else:
            print("\n❌ RESULTS DIFFER SIGNIFICANTLY!")
            
    # Compare regime performance
    if direct_perf and optimizer_perf:
        print("\nRegime Performance Comparison:")
        all_regimes = set(direct_perf.keys()) | set(optimizer_perf.keys())
        
        for regime in sorted(all_regimes):
            if not regime.startswith('_'):
                direct_trades = direct_perf.get(regime, {}).get('count', 0)
                optimizer_trades = optimizer_perf.get(regime, {}).get('count', 0)
                print(f"  {regime}:")
                print(f"    Direct trades: {direct_trades}")
                print(f"    Optimizer trades: {optimizer_trades}")
                
    print("=" * 80)


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config_path = "config/config_adaptive_production.yaml"
    params_file = "regime_optimized_parameters.json"
    
    if not Path(params_file).exists():
        logger.error(f"Parameters file not found: {params_file}")
        logger.error("Please run optimization first to generate regime parameters")
        sys.exit(1)
    
    try:
        config_loader = SimpleConfigLoader(config_file_path=config_path)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)
        
    setup_logging(config_loader, "INFO")
    
    # Create event bus and container
    event_bus = EventBus()
    container = setup_container(config_loader, event_bus)
    
    # Create backtest engine
    backtest_engine = BacktestEngine(container, config_loader, event_bus)
    
    # Run direct backtest
    print("\n1. Running direct backtest with BacktestEngine...")
    direct_metric, direct_perf = run_direct_backtest(backtest_engine, params_file)
    
    # Run through optimizer
    print("\n2. Running adaptive test through optimizer...")
    optimizer_metric, optimizer_perf = run_optimizer_adaptive_test(container, params_file)
    
    # Compare results
    compare_results(direct_metric, direct_perf, optimizer_metric, optimizer_perf)


if __name__ == "__main__":
    main()