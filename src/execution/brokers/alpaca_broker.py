#!/usr/bin/env python3
"""
Alpaca Broker Implementation - Connects to Alpaca Markets API.

Implements the BaseBroker interface for Alpaca's REST and WebSocket APIs.
"""

import os
import logging
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
import json
import threading
from queue import Queue
import time

from .base_broker import BaseBroker, OrderStatus, OrderType, OrderSide

# Note: In production, you'd install alpaca-py: pip install alpaca-py
try:
    from alpaca.trading.client import TradingClient
    from alpaca.data.live import StockDataStream
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logging.warning("alpaca-py not installed. Install with: pip install alpaca-py")


class AlpacaBroker(BaseBroker):
    """
    Alpaca broker implementation.
    
    Uses Alpaca's Python SDK for both REST and WebSocket connections.
    """
    
    def __init__(self, paper_trading: bool = True):
        """Initialize Alpaca broker."""
        super().__init__(paper_trading)
        self.logger = logging.getLogger(__name__)
        self.trading_client = None
        self.data_stream = None
        self.credentials = None
        self._bar_callbacks = {}
        self._reconnect_thread = None
        self._stop_reconnect = False
        
    def connect(self, credentials: Dict[str, str]) -> bool:
        """
        Connect to Alpaca API.
        
        Args:
            credentials: Must contain 'api_key' and 'secret_key'
        """
        if not ALPACA_AVAILABLE:
            self.logger.error("alpaca-py not installed")
            return False
            
        try:
            self.credentials = credentials
            
            # Use paper trading URL if enabled
            base_url = None
            if self.paper_trading:
                base_url = "https://paper-api.alpaca.markets"
                
            # Initialize trading client
            self.trading_client = TradingClient(
                api_key=credentials['api_key'],
                secret_key=credentials['secret_key'],
                paper=self.paper_trading
            )
            
            # Test connection by getting account info
            account = self.trading_client.get_account()
            self.logger.info(f"Connected to Alpaca ({'paper' if self.paper_trading else 'live'}) - "
                           f"Account: {account.account_number[:4]}***")
            
            # Initialize data stream for live bars
            self.data_stream = StockDataStream(
                api_key=credentials['api_key'],
                secret_key=credentials['secret_key'],
                url_override=base_url
            )
            
            self.connected = True
            
            # Start reconnection monitor
            self._start_reconnect_monitor()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {e}")
            self.connected = False
            return False
            
    def disconnect(self) -> None:
        """Disconnect from Alpaca API."""
        self._stop_reconnect = True
        
        if self.data_stream:
            try:
                self.data_stream.close()
            except:
                pass
                
        self.connected = False
        self.logger.info("Disconnected from Alpaca")
        
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        if not self.connected or not self.trading_client:
            return False
            
        try:
            # Ping the API
            self.trading_client.get_account()
            return True
        except:
            self.connected = False
            return False
            
    def subscribe_bars(self, symbol: str, timeframe: str, callback: Callable) -> bool:
        """Subscribe to real-time bar data."""
        if not self.connected:
            return False
            
        try:
            # Store callback
            self._bar_callbacks[symbol] = callback
            
            # Convert timeframe
            tf = self._convert_timeframe(timeframe)
            
            # Define handler for this symbol
            async def bar_handler(data):
                # Convert Alpaca bar to our format
                bar_dict = {
                    'timestamp': data.timestamp,
                    'symbol': data.symbol,
                    'open': float(data.open),
                    'high': float(data.high),
                    'low': float(data.low),
                    'close': float(data.close),
                    'volume': int(data.volume)
                }
                
                # Call the callback
                if symbol in self._bar_callbacks:
                    self._bar_callbacks[symbol](bar_dict)
                    
            # Subscribe to bars
            self.data_stream.subscribe_bars(bar_handler, symbol)
            
            # Start the stream in a separate thread
            threading.Thread(target=self._run_stream, daemon=True).start()
            
            self.logger.info(f"Subscribed to {timeframe} bars for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to bars: {e}")
            return False
            
    def unsubscribe_bars(self, symbol: str) -> None:
        """Unsubscribe from bar data."""
        if symbol in self._bar_callbacks:
            del self._bar_callbacks[symbol]
            
        if self.data_stream:
            self.data_stream.unsubscribe_bars(symbol)
            
    def get_latest_quote(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get latest quote for symbol."""
        if not self.connected:
            return None
            
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            
            # Create data client
            data_client = StockHistoricalDataClient(
                api_key=self.credentials['api_key'],
                secret_key=self.credentials['secret_key']
            )
            
            # Get latest quote
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = data_client.get_stock_latest_quote(request)
            quote = quotes[symbol]
            
            return {
                'bid': float(quote.bid_price),
                'ask': float(quote.ask_price),
                'last': float(quote.ask_price)  # Alpaca doesn't provide last in quote
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get quote: {e}")
            return None
            
    def submit_order(self,
                    symbol: str,
                    side: OrderSide,
                    quantity: int,
                    order_type: OrderType = OrderType.MARKET,
                    limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    time_in_force: str = "day") -> Optional[str]:
        """Submit an order to Alpaca."""
        if not self.connected:
            return None
            
        try:
            # Convert our enums to Alpaca's
            alpaca_side = AlpacaOrderSide.BUY if side == OrderSide.BUY else AlpacaOrderSide.SELL
            alpaca_tif = self._convert_time_in_force(time_in_force)
            
            # Create order request based on type
            if order_type == OrderType.MARKET:
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=alpaca_tif
                )
            elif order_type == OrderType.LIMIT:
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    limit_price=limit_price
                )
            else:
                self.logger.error(f"Order type {order_type} not yet implemented")
                return None
                
            # Submit order
            order = self.trading_client.submit_order(order_request)
            
            self.logger.info(f"Submitted {order_type.value} order: "
                           f"{side.value} {quantity} {symbol} @ "
                           f"{limit_price if limit_price else 'market'}")
            
            return order.id
            
        except Exception as e:
            self.logger.error(f"Failed to submit order: {e}")
            return None
            
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self.connected:
            return False
            
        try:
            self.trading_client.cancel_order_by_id(order_id)
            self.logger.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order: {e}")
            return False
            
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an order."""
        if not self.connected:
            return None
            
        try:
            order = self.trading_client.get_order_by_id(order_id)
            
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'side': order.side,
                'quantity': int(order.qty),
                'filled_qty': int(order.filled_qty or 0),
                'status': self._convert_order_status(order.status),
                'order_type': order.order_type,
                'limit_price': float(order.limit_price) if order.limit_price else None,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get order status: {e}")
            return None
            
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get list of all open orders."""
        if not self.connected:
            return []
            
        try:
            orders = self.trading_client.get_orders()
            
            return [
                {
                    'order_id': order.id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'quantity': int(order.qty),
                    'order_type': order.order_type,
                    'limit_price': float(order.limit_price) if order.limit_price else None,
                    'status': self._convert_order_status(order.status)
                }
                for order in orders
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get open orders: {e}")
            return []
            
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        if not self.connected:
            return []
            
        try:
            positions = self.trading_client.get_all_positions()
            
            return [
                {
                    'symbol': pos.symbol,
                    'quantity': int(pos.qty),
                    'avg_price': float(pos.avg_entry_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pnl': float(pos.unrealized_pl),
                    'side': 'long' if int(pos.qty) > 0 else 'short'
                }
                for pos in positions
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return []
            
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        if not self.connected:
            return {}
            
        try:
            account = self.trading_client.get_account()
            
            return {
                'account_number': account.account_number,
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'portfolio_value': float(account.portfolio_value),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            return {}
            
    def get_trade_history(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get historical trades within date range."""
        # TODO: Implement using Alpaca's portfolio history API
        return []
        
    # Options-specific methods
    
    def get_option_contracts(self, underlying_symbol: str, expiration_date: str) -> List[Dict[str, Any]]:
        """
        Get available option contracts for a symbol and expiration date.
        
        Args:
            underlying_symbol: The underlying symbol (e.g., "SPY")
            expiration_date: Date in YYYY-MM-DD format
            
        Returns:
            List of option contracts with strike prices and symbols
        """
        if not self.connected:
            return []
            
        try:
            # Note: As of 2024, Alpaca's options trading is in beta
            # You need to request access and use their options endpoints
            
            # This would use Alpaca's options chain endpoint
            # For now, returning empty as implementation depends on API access
            self.logger.warning("Options chain API not yet implemented - need Alpaca options access")
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get option contracts: {e}")
            return []
            
    def get_option_quote(self, option_symbol: str) -> Optional[Dict[str, float]]:
        """
        Get quote for a specific option contract.
        
        Args:
            option_symbol: OCC format option symbol
            
        Returns:
            Dictionary with bid, ask, last prices for the option
        """
        if not self.connected:
            return None
            
        try:
            # This would use Alpaca's options quote endpoint
            self.logger.warning("Options quote API not yet implemented")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get option quote: {e}")
            return None
        
    # Helper methods
    
    def _convert_timeframe(self, timeframe: str) -> TimeFrame:
        """Convert our timeframe string to Alpaca TimeFrame."""
        mapping = {
            '1min': TimeFrame.Minute,
            '5min': TimeFrame(5, 'Minute'),
            '15min': TimeFrame(15, 'Minute'),
            '1hour': TimeFrame.Hour,
            '1day': TimeFrame.Day
        }
        return mapping.get(timeframe, TimeFrame.Minute)
        
    def _convert_time_in_force(self, tif: str) -> TimeInForce:
        """Convert time in force string to Alpaca enum."""
        mapping = {
            'day': TimeInForce.DAY,
            'gtc': TimeInForce.GTC,
            'ioc': TimeInForce.IOC,
            'fok': TimeInForce.FOK
        }
        return mapping.get(tif.lower(), TimeInForce.DAY)
        
    def _convert_order_status(self, alpaca_status: str) -> OrderStatus:
        """Convert Alpaca order status to our enum."""
        mapping = {
            'new': OrderStatus.SUBMITTED,
            'partially_filled': OrderStatus.PARTIAL,
            'filled': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'expired': OrderStatus.EXPIRED,
            'rejected': OrderStatus.REJECTED,
            'pending_new': OrderStatus.PENDING
        }
        return mapping.get(alpaca_status, OrderStatus.PENDING)
        
    def _run_stream(self):
        """Run the data stream."""
        try:
            self.data_stream.run()
        except Exception as e:
            self.logger.error(f"Data stream error: {e}")
            
    def _start_reconnect_monitor(self):
        """Start monitoring connection and reconnect if needed."""
        def monitor():
            while not self._stop_reconnect:
                time.sleep(30)  # Check every 30 seconds
                
                if not self.is_connected() and self.credentials:
                    self.logger.warning("Connection lost, attempting to reconnect...")
                    self.connect(self.credentials)
                    
        self._reconnect_thread = threading.Thread(target=monitor, daemon=True)
        self._reconnect_thread.start()