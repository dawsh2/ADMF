#!/usr/bin/env python3
"""
Live Data Handler - Connects to broker API for real-time market data.

This handler receives live market data and publishes BAR events exactly like
CSVDataHandler, ensuring the same execution path for strategies.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

from ..core.component_base import ComponentBase
from ..core.event_bus import Event, EventType
from ..core.subscription_manager import SubscriptionManager
from ..execution.brokers.alpaca_broker import AlpacaBroker


class LiveDataHandler(ComponentBase):
    """
    Live data handler that connects to broker API for real-time data.
    
    Publishes the same BAR events as CSVDataHandler to ensure strategies
    work identically in backtest and live modes.
    """
    
    def __init__(self, instance_name="live_data_handler"):
        """Initialize with instance name."""
        super().__init__(instance_name)
        self._symbol = None
        self._broker_type = None
        self._broker = None
        self._timeframe = None
        self._subscription_manager = SubscriptionManager()
        
    def initialize(self, context: Dict[str, Any]):
        """Initialize from context."""
        super().initialize(context)
        
        # Get configuration
        config = self._config_loader.get(f'components.{self.instance_name}.config', {})
        self._symbol = config.get('symbol', 'SPY')
        self._broker_type = config.get('broker_type', 'alpaca')
        self._timeframe = config.get('timeframe', '1min')
        
        # Check if paper trading is enabled
        cli_args = context.get('metadata', {}).get('cli_args', {})
        self._paper_trading = cli_args.get('paper', True)
        
        self.logger.info(f"LiveDataHandler initialized for {self._symbol} on {self._broker_type}")
        self.logger.info(f"Paper trading: {self._paper_trading}")
        
    def start(self):
        """Start receiving live data."""
        super().start()
        
        # Create broker instance
        if self._broker_type == 'alpaca':
            self._broker = AlpacaBroker(paper_trading=self._paper_trading)
        else:
            raise ValueError(f"Unsupported broker type: {self._broker_type}")
            
        # Get credentials from environment or config
        credentials = self._get_broker_credentials()
        
        # Connect to broker
        if self._broker.connect(credentials):
            # Subscribe to live bars
            success = self._broker.subscribe_bars(
                symbol=self._symbol,
                timeframe=self._timeframe,
                callback=self._on_live_bar
            )
            
            if success:
                self.logger.info(f"Successfully subscribed to {self._symbol} {self._timeframe} bars")
            else:
                raise RuntimeError(f"Failed to subscribe to {self._symbol} bars")
        else:
            raise RuntimeError(f"Failed to connect to {self._broker_type}")
        
    def stop(self):
        """Stop receiving live data."""
        if self._broker:
            self._broker.unsubscribe_bars(self._symbol)
            self._broker.disconnect()
        self.logger.info("Disconnected from live data feed")
        super().stop()
        
    def _on_live_bar(self, bar_data: Dict[str, Any]):
        """
        Handle incoming live bar data from broker.
        
        Args:
            bar_data: Raw bar data from broker API
        """
        # Convert broker data to our BAR event format
        bar_event = Event(
            event_type=EventType.BAR,
            payload={
                'timestamp': bar_data['timestamp'],
                'symbol': self._symbol,
                'open': bar_data['open'],
                'high': bar_data['high'],
                'low': bar_data['low'],
                'close': bar_data['close'],
                'volume': bar_data['volume']
            }
        )
        
        # Publish BAR event - exactly like CSVDataHandler does
        self._event_bus.publish(bar_event)
        
        # Log for debugging
        self.logger.debug(f"Published BAR: {self._symbol} @ {bar_data['close']}")
        
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