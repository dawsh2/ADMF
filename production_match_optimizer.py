#!/usr/bin/env python3
"""
Option 1: Production that matches optimizer's data loading approach.
Loads FULL dataset, then processes only the test portion (bars 800-999).
"""

import sys
import json
import logging
from pathlib import Path

sys.path.append('.')

from src.core.config import SimpleConfigLoader
from src.core.container import Container
from src.core.event_bus import EventBus
from src.core.component import BaseComponent
from src.data.csv_data_handler import CSVDataHandler
from src.strategy.regime_detector import RegimeDetector
from src.strategy.regime_adaptive_strategy import RegimeAdaptiveStrategy
from src.risk.basic_portfolio import BasicPortfolio
from src.risk.basic_risk_manager import BasicRiskManager
from src.execution.simulated_execution_handler import SimulatedExecutionHandler
from src.core.logging_setup import setup_logging
from src.core.dummy_component import DummyComponent

def run_production_like_optimizer():
    """
    Run production exactly like the optimizer does:
    1. Load full dataset
    2. Switch to test portion
    3. Run backtest on bars 800-999
    """
    
    print("\n" + "="*80)
    print("OPTION 1: PRODUCTION MATCHING OPTIMIZER'S DATA LOADING")
    print("="*80)
    print("Loading full dataset, then using test portion (bars 800-999)")
    print("")
    
    # Load configuration
    config_loader = SimpleConfigLoader('config/config.yaml')
    event_bus = EventBus()
    container = Container()
    
    # Setup logging
    setup_logging(config_loader, 'ERROR')
    logger = logging.getLogger(__name__)
    
    # Register all components (matching main.py setup)
    logger.info("Setting up components...")
    
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
    
    # Strategy
    strategy_args = {
        "instance_name": "SPY_RegimeAdaptive_Strategy",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "container": container,
        "component_config_key": "components.regime_adaptive_strategy"
    }
    container.register_type("strategy", RegimeAdaptiveStrategy, True, constructor_kwargs=strategy_args)
    
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
    
    # Resolve components
    data_handler = container.resolve("data_handler")
    regime_detector = container.resolve("MyPrimaryRegimeDetector")
    portfolio_manager = container.resolve("portfolio_manager")
    risk_manager = container.resolve("risk_manager")
    execution_handler = container.resolve("execution_handler")
    strategy = container.resolve("strategy")
    signal_logger = container.resolve("signal_consumer")
    order_logger = container.resolve("order_consumer")
    
    # Component order (matching optimizer)
    components = [
        regime_detector,
        execution_handler,
        risk_manager,
        strategy,
        portfolio_manager,
        data_handler,
        signal_logger,
        order_logger
    ]
    
    try:
        # Setup all components
        logger.info("Setting up components...")
        for comp in components:
            if hasattr(comp, 'setup'):
                comp.setup()
                
        # Key difference: Set data to test BEFORE starting components
        logger.info("Setting data handler to test dataset...")
        data_handler.set_active_dataset("test")
        
        # Start all components
        logger.info("Starting components...")
        for comp in components:
            if hasattr(comp, 'start'):
                comp.start()
                
        # Wait for data handler to complete
        logger.info("Running backtest...")
        import time
        while data_handler.get_state() == BaseComponent.STATE_STARTED:
            time.sleep(0.1)
            
        # Get results
        final_value = portfolio_manager.get_final_portfolio_value()
        regime_performance = {}
        if hasattr(portfolio_manager, 'get_performance_by_regime'):
            regime_performance = portfolio_manager.get_performance_by_regime()
            
        # Display results
        print(f"\nFinal Portfolio Value: ${final_value:.2f}")
        
        if regime_performance:
            print("\nPERFORMANCE BY REGIME:")
            regimes_found = []
            for regime, metrics in regime_performance.items():
                if not regime.startswith('_'):
                    regimes_found.append(regime)
                    trades = metrics.get('count', 0)
                    sharpe = metrics.get('sharpe_ratio', 0)
                    print(f"  {regime}: {trades} trades, Sharpe: {sharpe:.4f}")
            print(f"\nRegimes detected: {', '.join(regimes_found)}")
            
    finally:
        # Stop all components
        logger.info("Stopping components...")
        for comp in reversed(components):
            if hasattr(comp, 'stop'):
                try:
                    comp.stop()
                except:
                    pass
                    
    print("="*80)
    print("This should match optimizer's result of $100,058.98")
    print("="*80)

if __name__ == "__main__":
    run_production_like_optimizer()