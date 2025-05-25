#!/usr/bin/env python3
"""
Verify that OOS test results match production run results exactly.

This script runs both the optimizer's adaptive test and a standalone production
backtest, then compares the results to ensure they match.
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

# Import components
from src.data.csv_data_handler import CSVDataHandler
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy
from src.strategy.regime_detector import RegimeDetector
from src.risk.basic_risk_manager import BasicRiskManager
from src.execution.simulated_execution_handler import SimulatedExecutionHandler
from src.risk.basic_portfolio import BasicPortfolio
from src.strategy.regime_adaptive_strategy import RegimeAdaptiveStrategy
from src.strategy.optimization.enhanced_optimizer_v2 import EnhancedOptimizerV2

logger = logging.getLogger(__name__)


def setup_container(config_loader, event_bus):
    """Set up container with all required components."""
    container = Container()
    container.register_instance("config_loader", config_loader)
    container.register_instance("event_bus", event_bus)
    container.register_instance("container", container)
    
    # Register all components
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


def run_optimizer_oos_test(container, params_file):
    """Run the optimizer's OOS adaptive test."""
    print("\n" + "="*80)
    print("RUNNING OPTIMIZER OOS TEST")
    print("="*80)
    
    # Load parameters to create mock results
    with open(params_file, 'r') as f:
        regime_params = json.load(f)
    
    # Create results structure
    results = {
        "best_parameters_per_regime": {},
        "best_metric_per_regime": {}
    }
    
    for regime, params in regime_params.items():
        results["best_parameters_per_regime"][regime] = {
            "parameters": params,
            "metric": {"name": "sharpe_ratio", "value": 0.0}
        }
    
    # Get optimizer
    optimizer = container.resolve("optimizer_service")
    optimizer.setup()
    
    # Set up data for optimizer
    data_handler = container.resolve("data_handler")
    data_handler.setup()
    data_handler.start()
    
    # Create mock train/test DataFrames
    if hasattr(data_handler, 'data_frame'):
        split_index = int(len(data_handler.data_frame) * 0.8)
        optimizer.train_df = data_handler.data_frame.iloc[:split_index]
        optimizer.test_df = data_handler.data_frame.iloc[split_index:]
    
    # Run adaptive test
    updated_results = optimizer.run_adaptive_test(results)
    
    metric_value = None
    regime_performance = None
    trades_count = None
    
    if "regime_adaptive_test_results" in updated_results:
        adaptive_results = updated_results["regime_adaptive_test_results"]
        metric_value = adaptive_results.get("adaptive_metric")
        regime_performance = adaptive_results.get("adaptive_regime_performance", {})
        
        # Count total trades
        trades_count = sum(
            perf.get('count', 0) 
            for regime, perf in regime_performance.items() 
            if not regime.startswith('_')
        )
    
    print(f"\nOptimizer OOS Test Results:")
    print(f"  Final Portfolio Value: {metric_value}")
    print(f"  Total Trades: {trades_count}")
    print(f"  Regimes: {list(regime_performance.keys()) if regime_performance else []}")
    
    return metric_value, regime_performance, trades_count


def run_production_backtest(container, config_loader, event_bus, params_file):
    """Run standalone production backtest."""
    print("\n" + "="*80)
    print("RUNNING PRODUCTION BACKTEST")
    print("="*80)
    
    # Create fresh BacktestEngine
    backtest_engine = BacktestEngine(container, config_loader, event_bus)
    
    # Run backtest
    metric_value, regime_performance = backtest_engine.run_backtest(
        parameters={},
        dataset_type="test",  # Use test dataset to match OOS
        strategy_type="regime_adaptive",
        use_regime_adaptive=True,
        adaptive_params_path=params_file
    )
    
    # Count trades
    trades_count = None
    if regime_performance:
        trades_count = sum(
            perf.get('count', 0) 
            for regime, perf in regime_performance.items() 
            if not regime.startswith('_')
        )
    
    print(f"\nProduction Backtest Results:")
    print(f"  Final Portfolio Value: {metric_value}")
    print(f"  Total Trades: {trades_count}")
    print(f"  Regimes: {list(regime_performance.keys()) if regime_performance else []}")
    
    return metric_value, regime_performance, trades_count


def compare_results(oos_metric, oos_perf, oos_trades, prod_metric, prod_perf, prod_trades):
    """Compare OOS test and production results."""
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    # Compare final values
    print(f"\nFinal Portfolio Values:")
    print(f"  OOS Test:    {oos_metric:.2f}" if oos_metric else "  OOS Test:    Failed")
    print(f"  Production:  {prod_metric:.2f}" if prod_metric else "  Production:  Failed")
    
    if oos_metric and prod_metric:
        difference = abs(oos_metric - prod_metric)
        pct_diff = (difference / oos_metric) * 100
        print(f"  Difference:  {difference:.2f} ({pct_diff:.4f}%)")
        
        if pct_diff < 0.01:
            print("\n✅ RESULTS MATCH! (difference < 0.01%)")
        else:
            print("\n❌ RESULTS DIFFER!")
    
    # Compare trade counts
    print(f"\nTotal Trades:")
    print(f"  OOS Test:    {oos_trades}")
    print(f"  Production:  {prod_trades}")
    
    # Compare regime performance
    if oos_perf and prod_perf:
        print("\nRegime Performance Comparison:")
        all_regimes = set(oos_perf.keys()) | set(prod_perf.keys())
        
        for regime in sorted(all_regimes):
            if not regime.startswith('_'):
                oos_data = oos_perf.get(regime, {})
                prod_data = prod_perf.get(regime, {})
                
                print(f"\n  {regime}:")
                print(f"    OOS trades:  {oos_data.get('count', 0)}")
                print(f"    Prod trades: {prod_data.get('count', 0)}")
                
                oos_sharpe = oos_data.get('sharpe_ratio', 'N/A')
                prod_sharpe = prod_data.get('sharpe_ratio', 'N/A')
                print(f"    OOS Sharpe:  {oos_sharpe}")
                print(f"    Prod Sharpe: {prod_sharpe}")
    
    print("="*80)


def main():
    # Configuration
    config_path = "config/config.yaml"
    params_file = "regime_optimized_parameters.json"
    
    # Check files exist
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
        
    if not Path(params_file).exists():
        logger.error(f"Parameters file not found: {params_file}")
        logger.error("Please run optimization first")
        sys.exit(1)
    
    # Load configuration
    try:
        config_loader = SimpleConfigLoader(config_file_path=config_path)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Set up logging
    setup_logging(config_loader, "INFO")
    
    # Create event bus and container
    event_bus = EventBus()
    container = setup_container(config_loader, event_bus)
    
    print("\nVERIFYING OOS TEST vs PRODUCTION MATCH")
    print("This will run both the optimizer's adaptive test and a standalone")
    print("production backtest, then compare the results.")
    
    # Run optimizer OOS test
    oos_metric, oos_perf, oos_trades = run_optimizer_oos_test(container, params_file)
    
    # Run production backtest
    prod_metric, prod_perf, prod_trades = run_production_backtest(
        container, config_loader, event_bus, params_file
    )
    
    # Compare results
    compare_results(oos_metric, oos_perf, oos_trades, prod_metric, prod_perf, prod_trades)
    
    # Provide recommendations if they don't match
    if oos_metric and prod_metric:
        difference = abs(oos_metric - prod_metric)
        pct_diff = (difference / oos_metric) * 100
        
        if pct_diff >= 0.01:
            print("\nPOSSIBLE CAUSES OF MISMATCH:")
            print("1. RegimeDetector internal indicators resetting (Part 1.2)")
            print("2. Different fallback parameter logic (Part 1.3)")
            print("3. Component initialization order differences")
            print("4. State persistence between runs")
            print("\nNext step: Run enhanced_signal_analysis to trace the differences")


if __name__ == "__main__":
    main()