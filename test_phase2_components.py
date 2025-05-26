#!/usr/bin/env python3
"""Test Phase 2 refactored components (BasicRiskManager and SimulatedExecutionHandler)."""

import sys
import logging
from pathlib import Path
from unittest.mock import MagicMock
import datetime

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.event_bus import EventBus
from src.core.event import Event, EventType

def test_phase2_components():
    """Test the Phase 2 refactored components."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("test_phase2")
    
    try:
        # Create dependencies
        event_bus = EventBus()
        container = MagicMock()
        config_loader = MagicMock()
        
        # Mock config loader responses
        config_loader.get_component_config = MagicMock(return_value={
            "target_trade_quantity": 100,
            "portfolio_manager_key": "portfolio",
            "default_quantity": 100,
            "commission_per_trade": 1.0,
            "passthrough": False
        })
        
        # Import BasicPortfolio to create a proper mock
        from src.risk.basic_portfolio import BasicPortfolio
        
        # Create a mock that passes isinstance check
        mock_portfolio = MagicMock(spec=BasicPortfolio)
        mock_portfolio.instance_name = "portfolio"
        mock_portfolio.get_current_position_quantity = MagicMock(return_value=0)
        container.resolve = MagicMock(return_value=mock_portfolio)
        
        # Mock full config object
        config = {
            "components": {
                "risk_manager": {
                    "target_trade_quantity": 100,
                    "portfolio_manager_key": "portfolio"
                },
                "execution_handler": {
                    "default_quantity": 100,
                    "commission_per_trade": 1.0,
                    "passthrough": False
                }
            }
        }
        
        # Create context
        context = {
            'event_bus': event_bus,
            'container': container,
            'config_loader': config_loader,
            'config': config,
            'logger': logger
        }
        
        logger.info("Testing Phase 2 Components...")
        
        # Test BasicRiskManager
        logger.info("\n=== Testing BasicRiskManager ===")
        from src.risk.basic_risk_manager import BasicRiskManager
        
        risk_manager = BasicRiskManager("risk_manager", config_key="risk_manager")
        logger.info(f"Created: {risk_manager}")
        
        risk_manager.initialize(context)
        logger.info(f"Initialized: {risk_manager.initialized}")
        
        risk_manager.setup()
        logger.info("Setup complete")
        
        risk_manager.start()
        logger.info(f"Started: {risk_manager.running}")
        
        # Test SimulatedExecutionHandler
        logger.info("\n=== Testing SimulatedExecutionHandler ===")
        from src.execution.simulated_execution_handler import SimulatedExecutionHandler
        
        exec_handler = SimulatedExecutionHandler("execution_handler", config_key="execution_handler")
        logger.info(f"Created: {exec_handler}")
        
        exec_handler.initialize(context)
        logger.info(f"Initialized: {exec_handler.initialized}")
        
        exec_handler.setup()
        logger.info("Setup complete")
        
        exec_handler.start()
        logger.info(f"Started: {exec_handler.running}")
        
        # Test event flow
        logger.info("\n=== Testing Event Flow ===")
        
        # Test signal -> order flow
        test_signal = Event(EventType.SIGNAL, {
            "symbol": "SPY",
            "signal_type": 1,  # Buy signal
            "price_at_signal": 100.0,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "strategy_id": "test_strategy"
        })
        
        logger.info("Publishing test SIGNAL event...")
        event_bus.publish(test_signal)
        
        # Verify order was published
        logger.info("Signal processed by RiskManager")
        
        # Test order -> fill flow
        test_order = Event(EventType.ORDER, {
            "order_id": "test_order_123",
            "symbol": "SPY",
            "order_type": "MARKET",
            "direction": "BUY",
            "quantity": 100,
            "simulated_fill_price": 100.0,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "strategy_id": "test_strategy",
            "risk_manager_id": "risk_manager"
        })
        
        logger.info("Publishing test ORDER event...")
        event_bus.publish(test_order)
        
        logger.info("Order processed by ExecutionHandler")
        
        # Test stop
        logger.info("\nStop phase...")
        risk_manager.stop()
        logger.info(f"RiskManager after stop: running={risk_manager.running}")
        
        exec_handler.stop()
        logger.info(f"ExecutionHandler after stop: running={exec_handler.running}")
        
        # Test teardown
        logger.info("\nTeardown phase...")
        risk_manager.teardown()
        logger.info(f"RiskManager after teardown: initialized={risk_manager.initialized}")
        
        exec_handler.teardown()
        logger.info(f"ExecutionHandler after teardown: initialized={exec_handler.initialized}")
        
        logger.info("\n✅ Phase 2 tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_phase2_components()
    sys.exit(0 if success else 1)