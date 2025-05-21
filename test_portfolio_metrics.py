#!/usr/bin/env python3
# test_portfolio_metrics.py
import logging
import sys
from typing import Dict, Any

from src.core.config import SimpleConfigLoader
from src.core.logging_setup import setup_logging
from src.core.event_bus import EventBus
from src.core.container import Container
from src.risk.basic_portfolio import BasicPortfolio

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
    logger = logging.getLogger("test_portfolio_metrics")
    
    # Load configuration
    config_loader = SimpleConfigLoader(config_file_path="config/config.yaml")
    setup_logging(config_loader)
    
    # Create components
    event_bus = EventBus()
    container = Container()
    
    # Initialize portfolio
    portfolio = BasicPortfolio(
        instance_name="TestPortfolio",
        config_loader=config_loader,
        event_bus=event_bus,
        container=container,
        component_config_key="components.basic_portfolio"
    )
    
    # Setup portfolio
    portfolio.setup()
    
    # Inspect available methods
    logger.info("Available portfolio methods:")
    methods = [m for m in dir(portfolio) if callable(getattr(portfolio, m)) and not m.startswith('_')]
    for method in methods:
        logger.info(f"  - {method}")
    
    # Test get_performance method if it exists
    if hasattr(portfolio, 'get_performance') and callable(getattr(portfolio, 'get_performance')):
        performance = portfolio.get_performance()
        logger.info(f"Portfolio performance: {performance}")
    else:
        logger.warning("get_performance method not found")
    
    # Test get_final_portfolio_value method if it exists
    if hasattr(portfolio, 'get_final_portfolio_value') and callable(getattr(portfolio, 'get_final_portfolio_value')):
        value = portfolio.get_final_portfolio_value()
        logger.info(f"Final portfolio value: {value}")
    else:
        logger.warning("get_final_portfolio_value method not found")
        
    # Check current_total_value
    logger.info(f"Portfolio current_total_value: {portfolio.current_total_value}")
    
    # Check available attributes
    logger.info("Available portfolio attributes:")
    attributes = [a for a in dir(portfolio) if not callable(getattr(portfolio, a)) and not a.startswith('_')]
    for attr in attributes:
        logger.info(f"  - {attr}: {getattr(portfolio, attr)}")

if __name__ == "__main__":
    main()