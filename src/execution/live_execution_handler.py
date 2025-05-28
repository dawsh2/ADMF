#!/usr/bin/env python3
"""
Live Execution Handler - Submits orders to broker and monitors fills.

This handler receives ORDER events and submits them to the broker API,
then publishes FILL events exactly like SimulatedExecutionHandler.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

from ..core.component_base import ComponentBase
from ..core.event_bus import Event, EventType
from ..core.subscription_manager import SubscriptionManager
from .brokers.alpaca_broker import AlpacaBroker
from .brokers.base_broker import OrderSide, OrderType


class LiveExecutionHandler(ComponentBase):
    """
    Live execution handler that submits orders to broker API.
    
    Receives ORDER events and publishes FILL events exactly like
    SimulatedExecutionHandler to ensure strategies work identically.
    """
    
    def __init__(self, instance_name="live_execution_handler"):
        """Initialize with instance name."""
        super().__init__(instance_name)
        self._broker_type = None
        self._broker = None
        self._paper_trading = True
        self._pending_orders = {}
        self._subscription_manager = SubscriptionManager()
        self._order_check_interval = 1.0  # Check order status every second
        
    def initialize(self, context: Dict[str, Any]):
        """Initialize from context."""
        super().initialize(context)
        
        # Get configuration
        config = self._config_loader.get(f'components.{self.instance_name}.config', {})
        self._broker_type = config.get('broker_type', 'alpaca')
        
        # Check if paper trading is enabled
        cli_args = context.get('metadata', {}).get('cli_args', {})
        self._paper_trading = cli_args.get('paper', True)
        
        self.logger.info(f"LiveExecutionHandler initialized for {self._broker_type}")
        self.logger.info(f"Paper trading: {self._paper_trading}")
        
    def initialize_event_subscriptions(self):
        """Subscribe to ORDER events."""
        self._subscription_manager.subscribe(
            event_type=EventType.ORDER,
            handler=self._on_order_event,
            context=self
        )
        
    def start(self):
        """Start order execution."""
        super().start()
        
        # Create broker instance
        if self._broker_type == 'alpaca':
            self._broker = AlpacaBroker(paper_trading=self._paper_trading)
        else:
            raise ValueError(f"Unsupported broker type: {self._broker_type}")
            
        # Get credentials
        credentials = self._get_broker_credentials()
        
        # Connect to broker
        if self._broker.connect(credentials):
            self.logger.info(f"Connected to {self._broker_type} broker")
            if self._paper_trading:
                self.logger.info("Using paper trading account")
            else:
                self.logger.warning("Using LIVE trading account - be careful!")
                
            # Start order monitoring thread
            self._start_order_monitoring()
            
            # Reconcile positions on startup
            self._reconcile_positions()
        else:
            raise RuntimeError(f"Failed to connect to {self._broker_type}")
            
    def stop(self):
        """Stop order execution."""
        if self._broker:
            # Cancel all pending orders
            for order_id in list(self._pending_orders.keys()):
                self._broker.cancel_order(order_id)
            
            # Disconnect from broker
            self._broker.disconnect()
            
        self.logger.info("Disconnected from broker API")
        super().stop()
        
    def _on_order_event(self, event: Event):
        """
        Handle incoming ORDER events.
        
        Args:
            event: ORDER event from risk manager
        """
        order = event.payload
        self.logger.info(f"Received ORDER: {order['side']} {order['quantity']} {order['symbol']}")
        
        # Convert order to broker format
        side = OrderSide.BUY if order['side'] == 'buy' else OrderSide.SELL
        
        # Submit order to broker
        broker_order_id = self._broker.submit_order(
            symbol=order['symbol'],
            side=side,
            quantity=order['quantity'],
            order_type=OrderType.MARKET,  # Using market orders for now
            limit_price=None,
            time_in_force='day'
        )
        
        if broker_order_id:
            # Store order for monitoring
            self._pending_orders[broker_order_id] = order
            self.logger.info(f"Order submitted to broker: {broker_order_id}")
        else:
            self.logger.error(f"Failed to submit order for {order['symbol']}")
            return
            
    def _get_broker_credentials(self) -> Dict[str, str]:
        """Get broker API credentials from environment or config."""
        import os
        
        # First try environment variables
        if self._broker_type == 'alpaca':
            api_key = os.environ.get('ALPACA_API_KEY')
            secret_key = os.environ.get('ALPACA_SECRET_KEY')
            
            if api_key and secret_key:
                return {
                    'api_key': api_key,
                    'secret_key': secret_key
                }
                
        # Try config file
        broker_config = self._config_loader.get('broker', {})
        if 'credentials' in broker_config:
            return broker_config['credentials']
            
        raise ValueError(f"No credentials found for {self._broker_type}. "
                        "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
                        
    def _start_order_monitoring(self):
        """Start monitoring pending orders for fills."""
        import threading
        
        def monitor_orders():
            while self.running:
                # Check each pending order
                for order_id, original_order in list(self._pending_orders.items()):
                    status = self._broker.get_order_status(order_id)
                    
                    if status:
                        if status['status'].value == 'filled':
                            # Create FILL event
                            self._publish_fill_event(original_order, status)
                            # Remove from pending
                            del self._pending_orders[order_id]
                        elif status['status'].value in ['cancelled', 'rejected']:
                            self.logger.warning(f"Order {order_id} was {status['status'].value}")
                            del self._pending_orders[order_id]
                            
                # Sleep before next check
                import time
                time.sleep(self._order_check_interval)
                
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_orders, daemon=True)
        monitor_thread.start()
        
    def _reconcile_positions(self):
        """Reconcile positions with broker on startup."""
        positions = self._broker.get_positions()
        if positions:
            self.logger.info(f"Current broker positions: {len(positions)}")
            for pos in positions:
                self.logger.info(f"  {pos['symbol']}: {pos['quantity']} @ {pos['avg_price']}")
        else:
            self.logger.info("No open positions at broker")
            
    def _publish_fill_event(self, original_order: Dict[str, Any], broker_status: Dict[str, Any]):
        """Publish FILL event from broker order status."""
        fill_event = Event(
            event_type=EventType.FILL,
            payload={
                'timestamp': broker_status.get('filled_at', datetime.now()),
                'symbol': original_order['symbol'],
                'side': original_order['side'],
                'quantity': broker_status.get('filled_qty', original_order['quantity']),
                'price': broker_status.get('filled_avg_price', original_order.get('price', 0)),
                'commission': 0.0,  # TODO: Calculate from broker
                'order_id': original_order.get('order_id')
            }
        )
        
        self._event_bus.publish(fill_event)
        self.logger.info(f"Order filled: {original_order['side']} {broker_status['filled_qty']} "
                        f"@ {broker_status['filled_avg_price']}")