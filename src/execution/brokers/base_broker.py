#!/usr/bin/env python3
"""
Base Broker Interface - Abstract interface for all broker connections.

Defines the contract that all broker implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
from enum import Enum


class OrderStatus(Enum):
    """Standard order statuses across all brokers."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Standard order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class BaseBroker(ABC):
    """
    Abstract base class for broker connections.
    
    All broker implementations must inherit from this class and implement
    the required methods to ensure consistent behavior across different brokers.
    """
    
    def __init__(self, paper_trading: bool = True):
        """
        Initialize broker connection.
        
        Args:
            paper_trading: If True, use paper trading account
        """
        self.paper_trading = paper_trading
        self.connected = False
        
    @abstractmethod
    def connect(self, credentials: Dict[str, str]) -> bool:
        """
        Connect to broker API.
        
        Args:
            credentials: Dictionary containing API keys and endpoints
            
        Returns:
            True if connection successful, False otherwise
        """
        pass
        
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker API."""
        pass
        
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        pass
        
    # Market Data Methods
    
    @abstractmethod
    def subscribe_bars(self, symbol: str, timeframe: str, callback: Callable) -> bool:
        """
        Subscribe to real-time bar data.
        
        Args:
            symbol: Symbol to subscribe to (e.g., "SPY")
            timeframe: Bar timeframe (e.g., "1min", "5min")
            callback: Function to call with bar data
            
        Returns:
            True if subscription successful
        """
        pass
        
    @abstractmethod
    def unsubscribe_bars(self, symbol: str) -> None:
        """Unsubscribe from bar data."""
        pass
        
    @abstractmethod
    def get_latest_quote(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get latest quote for symbol.
        
        Returns:
            Dictionary with 'bid', 'ask', 'last' prices
        """
        pass
        
    # Order Management Methods
    
    @abstractmethod
    def submit_order(self, 
                    symbol: str,
                    side: OrderSide,
                    quantity: int,
                    order_type: OrderType = OrderType.MARKET,
                    limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    time_in_force: str = "day") -> Optional[str]:
        """
        Submit an order to the broker.
        
        Args:
            symbol: Symbol to trade
            side: Buy or sell
            quantity: Number of shares
            order_type: Type of order
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Order time in force (day, gtc, ioc, fok)
            
        Returns:
            Order ID if successful, None otherwise
        """
        pass
        
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancellation successful
        """
        pass
        
    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of an order.
        
        Returns:
            Dictionary with order details and status
        """
        pass
        
    @abstractmethod
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get list of all open orders."""
        pass
        
    # Account Methods
    
    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
            List of position dictionaries with 'symbol', 'quantity', 'avg_price'
        """
        pass
        
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dictionary with 'cash', 'buying_power', 'equity' etc.
        """
        pass
        
    @abstractmethod
    def get_trade_history(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get historical trades within date range."""
        pass