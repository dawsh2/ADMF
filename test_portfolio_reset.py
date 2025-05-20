#!/usr/bin/env python3
# test_portfolio_reset.py
"""
A simple test script to verify that the portfolio reset functionality works correctly.
This script performs two consecutive backtest runs with identical parameters
and verifies that each run starts with the same initial state.
"""

import logging
import datetime
import sys
from typing import Dict, Any, List

from src.core.config import SimpleConfigLoader
from src.core.container import Container
from src.core.event_bus import EventBus
from src.data.csv_data_handler import CSVDataHandler
from src.strategy.ma_strategy import MAStrategy
from src.risk.basic_portfolio import BasicPortfolio
from src.execution.simulated_execution_handler import SimulatedExecutionHandler
from src.risk.basic_risk_manager import BasicRiskManager

def setup_logging():
    """Configure logging for the test."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("test_portfolio_reset.log")
        ]
    )
    return logging.getLogger("PortfolioResetTest")

def create_components(config_loader, container, event_bus):
    """Create and register all required components."""
    # Create data handler with proper configuration
    data_handler = CSVDataHandler(
        "DataHandler", config_loader, event_bus, "data_handler_csv"
    )
    container.register_instance("data_handler", data_handler)
    
    # Create portfolio manager
    portfolio = BasicPortfolio(
        "Portfolio", config_loader, event_bus, container, "basic_portfolio"
    )
    container.register_instance("portfolio_manager", portfolio)
    
    # Create strategy
    strategy = MAStrategy(
        "Strategy", config_loader, event_bus, container, "ensemble_strategy"
    )
    container.register_instance("strategy", strategy)
    
    # Create risk manager
    risk_manager = BasicRiskManager(
        "RiskManager", config_loader, event_bus, "basic_risk_manager", container
    )
    container.register_instance("risk_manager", risk_manager)
    
    # Create execution handler
    execution_handler = SimulatedExecutionHandler(
        "ExecutionHandler", config_loader, event_bus, "simulated_execution_handler"
    )
    container.register_instance("execution_handler", execution_handler)
    
    return [data_handler, portfolio, strategy, risk_manager, execution_handler]

def run_backtest(components, logger):
    """Run a single backtest and return the results."""
    data_handler, portfolio, strategy, risk_manager, execution_handler = components
    
    # Setup all components
    for comp in components:
        comp.setup()
        if comp.get_state() == "FAILED":
            logger.error(f"Component {comp.name} failed setup")
            return None
    
    # Log initial portfolio value
    logger.info(f"Initial portfolio cash: {portfolio.current_cash:.2f}")
    logger.info(f"Initial portfolio value: {portfolio.current_total_value:.2f}")
    
    # Start all components
    for comp in components:
        comp.start()
        if comp.get_state() == "FAILED":
            logger.error(f"Component {comp.name} failed to start")
            return None
    
    # Close all positions at the end of the run
    last_timestamp = data_handler.get_last_timestamp() or datetime.datetime.now(datetime.timezone.utc)
    portfolio.close_all_positions(last_timestamp)
    
    # Log final portfolio value
    logger.info(f"Final portfolio cash: {portfolio.current_cash:.2f}")
    logger.info(f"Final portfolio value: {portfolio.current_total_value:.2f}")
    logger.info(f"Realized PnL: {portfolio.realized_pnl:.2f}")
    logger.info(f"Number of trades: {len(portfolio._trade_log)}")
    
    # Get final portfolio value
    final_value = portfolio.get_final_portfolio_value()
    
    # Stop all components in reverse order
    for comp in reversed(components):
        comp.stop()
    
    return final_value

def main():
    """Main test function."""
    logger = setup_logging()
    logger.info("Starting portfolio reset test")
    
    # Initialize core components
    config_loader = SimpleConfigLoader("config/config.yaml")
    container = Container()
    event_bus = EventBus()
    
    # Register core components
    container.register_instance("config", config_loader)
    container.register_instance("event_bus", event_bus)
    container.register_instance("container", container)
    
    # Create all trading components
    components = create_components(config_loader, container, event_bus)
    data_handler, portfolio, strategy, risk_manager, execution_handler = components
    
    # Run first backtest
    logger.info("=== STARTING FIRST BACKTEST RUN ===")
    first_run_value = run_backtest(components, logger)
    logger.info(f"First run final portfolio value: {first_run_value:.2f}")
    
    # Check portfolio state before reset
    logger.info("=== CHECKING PORTFOLIO STATE BEFORE RESET ===")
    logger.info(f"Portfolio cash after first run: {portfolio.current_cash:.2f}")
    logger.info(f"Portfolio value after first run: {portfolio.current_total_value:.2f}")
    
    # Reset portfolio
    logger.info("=== RESETTING PORTFOLIO ===")
    portfolio.reset()
    
    # Check portfolio state after reset
    logger.info("=== CHECKING PORTFOLIO STATE AFTER RESET ===")
    logger.info(f"Portfolio cash after reset: {portfolio.current_cash:.2f}")
    logger.info(f"Portfolio value after reset: {portfolio.current_total_value:.2f}")
    
    # Run second backtest with same parameters
    logger.info("=== STARTING SECOND BACKTEST RUN ===")
    second_run_value = run_backtest(components, logger)
    logger.info(f"Second run final portfolio value: {second_run_value:.2f}")
    
    # Compare results
    logger.info("=== COMPARISON OF RESULTS ===")
    logger.info(f"First run final value: {first_run_value:.2f}")
    logger.info(f"Second run final value: {second_run_value:.2f}")
    logger.info(f"Difference: {second_run_value - first_run_value:.2f}")
    
    # Determine if test passed
    if abs(first_run_value - second_run_value) < 0.01:
        logger.info("TEST RESULT: PASSED - Both runs produced identical results")
    else:
        logger.info("TEST RESULT: FAILED - Runs produced different results")
    
    logger.info("Portfolio reset test completed")

if __name__ == "__main__":
    main()