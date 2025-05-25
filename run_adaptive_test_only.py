#!/usr/bin/env python3
"""
Run ONLY the adaptive test without grid search optimization.
This tests if state accumulation during grid search is causing the difference.
"""

import sys
import json
import logging

sys.path.append('.')

from src.core.config import SimpleConfigLoader
from src.core.container import Container
from src.core.event_bus import EventBus
from src.strategy.optimization.enhanced_optimizer_v2 import EnhancedOptimizerV2
from src.core.logging_setup import setup_logging

# Import all required components for registration
from src.data.csv_data_handler import CSVDataHandler
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy
from src.strategy.regime_detector import RegimeDetector
from src.strategy.regime_adaptive_strategy import RegimeAdaptiveStrategy
from src.risk.basic_portfolio import BasicPortfolio
from src.risk.basic_risk_manager import BasicRiskManager
from src.execution.simulated_execution_handler import SimulatedExecutionHandler
from src.core.dummy_component import DummyComponent

def setup_container(config_loader, event_bus, container):
    """Set up the DI container with all required components (matching main.py)."""
    
    # Data handler
    csv_args = {
        "instance_name": "SPY_CSV_Loader",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.data_handler_csv"
    }
    container.register_type("data_handler", CSVDataHandler, True, constructor_kwargs=csv_args)
    
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
    
    # Strategies
    ensemble_args = {
        "instance_name": "SPY_Ensemble_Strategy",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.ensemble_strategy",
        "container": container
    }
    container.register_type("strategy", EnsembleSignalStrategy, True, constructor_kwargs=ensemble_args)
    
    # Loggers
    signal_logger_args = {
        "instance_name": "SignalLogger",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.dummy_service",
        "listen_to_event_type_str": "SIGNAL"
    }
    container.register_type("signal_consumer", DummyComponent, True, constructor_kwargs=signal_logger_args)
    
    order_logger_args = {
        "instance_name": "OrderLogger",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.dummy_service",
        "listen_to_event_type_str": "ORDER"
    }
    container.register_type("order_consumer", DummyComponent, True, constructor_kwargs=order_logger_args)

def run_adaptive_test_only():
    """Run only the adaptive test without any grid search."""
    
    print("\n" + "="*80)
    print("ADAPTIVE TEST ONLY (NO GRID SEARCH)")
    print("="*80)
    print("This runs ONLY the adaptive test, skipping all optimization.")
    print("Should produce $99,870 if state accumulation is the issue.")
    print("")
    
    # Load configuration
    config_loader = SimpleConfigLoader('config/config.yaml')
    event_bus = EventBus()
    container = Container()
    
    # Setup logging
    setup_logging(config_loader, 'ERROR')
    logger = logging.getLogger(__name__)
    
    # Register all components
    setup_container(config_loader, event_bus, container)
    
    # Create optimizer (V2)
    optimizer_args = {
        "instance_name": "EnhancedOptimizer",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.optimizer",
        "container": container
    }
    optimizer = EnhancedOptimizerV2(**optimizer_args)
    
    # Setup optimizer
    optimizer.setup()
    
    # Create fake results with pre-optimized parameters
    # (normally these would come from grid search)
    fake_results = {
        'best_parameters_per_regime': {}
    }
    
    # Load the optimized parameters
    with open('regime_optimized_parameters.json', 'r') as f:
        regime_params = json.load(f)
        
    # Format them as the optimizer expects
    for regime, params in regime_params.items():
        fake_results['best_parameters_per_regime'][regime] = {
            'parameters': params,
            'sharpe_ratio': 0.0  # Dummy value
        }
    
    # Run ONLY the adaptive test
    print("Running adaptive test with pre-loaded parameters...")
    results = optimizer.run_adaptive_test(fake_results)
    
    # Extract results
    if 'regime_adaptive_test_results' in results:
        adaptive_results = results['regime_adaptive_test_results']
        final_value = adaptive_results.get('adaptive_metric', 'N/A')
        regimes = adaptive_results.get('regimes_detected', [])
        
        print(f"\nFinal Portfolio Value: ${final_value:.2f}")
        print(f"Regimes detected: {', '.join(regimes)}")
    else:
        print("\nNo adaptive test results found!")
        
    print("\n" + "="*80)
    print("EXPECTED: $99,870.04 (matching production)")
    print("ACTUAL: See above")
    print("")
    print("If this matches production, it confirms state accumulation")
    print("during grid search is causing the $100,058.98 result.")
    print("="*80)

if __name__ == "__main__":
    run_adaptive_test_only()