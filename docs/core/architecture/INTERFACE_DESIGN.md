# Interface-Based Module Boundaries

## Overview

This document defines formal interfaces for cross-module interactions in the ADMF-Trader system. These interfaces establish clear boundaries between modules, enforce contracts, and enable proper dependency injection. The design follows the principles established in the enhanced optimization module, applying similar patterns system-wide.

## Interface Design Principles

1. **Explicit Contracts**: Interfaces explicitly define the contracts between components
2. **Dependency Inversion**: High-level modules depend on abstractions, not concrete implementations
3. **Isolation**: Interfaces enable module isolation for better testing and maintainability
4. **Extensibility**: New implementations can be created without modifying existing code
5. **Configuration-Driven**: Component selection and configuration via external configuration
6. **Type Safety**: Interfaces define clear type expectations
7. **Execution Mode Agnostic**: Interfaces support both synchronous and asynchronous implementations

> **Note on Asynchronous Support**: All interfaces defined in this document can be implemented in both synchronous and asynchronous forms. For details on asynchronous implementations, see [ASYNCHRONOUS_ARCHITECTURE.md](/Users/daws/ADMF/docs/core/ASYNCHRONOUS_ARCHITECTURE.md)

## Core Interfaces

### ComponentBase

The foundational interface that all components implement:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class ComponentBase(ABC):
    """Base interface for all system components."""
    
    @abstractmethod
    def initialize(self, context: Dict[str, Any]) -> None:
        """
        Initialize component with dependencies from context.
        
        Args:
            context: Dependency context containing required components
        """
        pass
    
    @abstractmethod
    def start(self) -> None:
        """Begin component operation."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """End component operation."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Clear component state for a new run."""
        pass
    
    @abstractmethod
    def teardown(self) -> None:
        """Release resources."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get component name."""
        pass
    
    @property
    @abstractmethod
    def initialized(self) -> bool:
        """Whether component is initialized."""
        pass
    
    @property
    @abstractmethod
    def running(self) -> bool:
        """Whether component is running."""
        pass
```

## Data Module Interfaces

### DataHandlerBase

Interface for data handling components:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import pandas as pd

class DataHandlerBase(ComponentBase):
    """Base interface for data handling components."""
    
    @abstractmethod
    def load_data(self, symbols: List[str], **kwargs) -> None:
        """
        Load data for specified symbols.
        
        Args:
            symbols: List of symbols to load
            **kwargs: Additional parameters for loading
        """
        pass
    
    @abstractmethod
    def update_bars(self) -> bool:
        """
        Update bars and emit events.
        
        Returns:
            bool: Whether more bars are available
        """
        pass
    
    @abstractmethod
    def get_latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest bar for a symbol.
        
        Args:
            symbol: Symbol to get bar for
            
        Returns:
            Dict containing bar data or None if not available
        """
        pass
    
    @abstractmethod
    def get_latest_bars(self, symbol: str, N: int = 1) -> List[Dict[str, Any]]:
        """
        Get the last N bars for a symbol.
        
        Args:
            symbol: Symbol to get bars for
            N: Number of bars to retrieve
            
        Returns:
            List of bar data dicts
        """
        pass
    
    @abstractmethod
    def get_bar_history(self, symbol: str, start: Optional[datetime] = None, 
                       end: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get historical bars for a symbol between start and end times.
        
        Args:
            symbol: Symbol to get history for
            start: Start time (None for earliest available)
            end: End time (None for latest available)
            
        Returns:
            List of bar data dicts
        """
        pass
    
    @abstractmethod
    def setup_train_test_split(self, method: str = 'ratio', train_ratio: float = 0.7, 
                              split_date: Optional[datetime] = None) -> None:
        """
        Set up training and testing data splits.
        
        Args:
            method: Split method ('ratio', 'date', or 'periods')
            train_ratio: Ratio of data for training (if method='ratio')
            split_date: Date to split at (if method='date')
        """
        pass
    
    @abstractmethod
    def set_active_split(self, split_name: str) -> None:
        """
        Set the active data split.
        
        Args:
            split_name: Name of split to activate ('train' or 'test')
        """
        pass
    
    @property
    @abstractmethod
    def symbols(self) -> List[str]:
        """Get list of available symbols."""
        pass
    
    @property
    @abstractmethod
    def timeframe(self) -> str:
        """Get data timeframe."""
        pass
    
    @property
    @abstractmethod
    def start_date(self) -> datetime:
        """Get start date of data."""
        pass
    
    @property
    @abstractmethod
    def end_date(self) -> datetime:
        """Get end date of data."""
        pass
```

### DataSourceBase

Interface for data source components:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import pandas as pd

class DataSourceBase(ComponentBase):
    """Base interface for data sources."""
    
    @abstractmethod
    def load(self, symbols: List[str], start_date: Optional[datetime] = None, 
            end_date: Optional[datetime] = None, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Load data for specified symbols.
        
        Args:
            symbols: List of symbols to load
            start_date: Start date for data (None for earliest available)
            end_date: End date for data (None for latest available)
            **kwargs: Additional parameters for loading
            
        Returns:
            Dict mapping symbols to DataFrames
        """
        pass
    
    @abstractmethod
    def get_symbols(self) -> List[str]:
        """
        Get available symbols from this data source.
        
        Returns:
            List of available symbols
        """
        pass
    
    @abstractmethod
    def get_timeframes(self) -> List[str]:
        """
        Get available timeframes from this data source.
        
        Returns:
            List of available timeframes
        """
        pass
    
    @property
    @abstractmethod
    def source_type(self) -> str:
        """Get data source type (e.g., 'csv', 'api')."""
        pass
```

## Strategy Module Interfaces

### StrategyBase

Interface for strategy components:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

class StrategyBase(ComponentBase):
    """Base interface for trading strategies."""
    
    @abstractmethod
    def on_bar(self, event: Dict[str, Any]) -> None:
        """
        Process bar event.
        
        Args:
            event: Bar event data
        """
        pass
    
    @abstractmethod
    def calculate_signals(self, bar_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Calculate signals based on bar data.
        
        Args:
            bar_data: Bar data dictionary
            
        Returns:
            List of signal dictionaries
        """
        pass
    
    @abstractmethod
    def emit_signal(self, symbol: str, direction: str, quantity: float, 
                   price: float, **kwargs) -> bool:
        """
        Emit a signal event.
        
        Args:
            symbol: Instrument symbol
            direction: Signal direction ('BUY' or 'SELL')
            quantity: Signal quantity
            price: Signal price
            **kwargs: Additional signal parameters
            
        Returns:
            bool: Whether signal was successfully emitted
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get strategy parameters.
        
        Returns:
            Dict of parameter names to values
        """
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set strategy parameters.
        
        Args:
            parameters: Dict of parameter names to values
        """
        pass
    
    @property
    @abstractmethod
    def parameter_space(self) -> Dict[str, List[Any]]:
        """Get strategy parameter space for optimization."""
        pass
```

### IndicatorBase

Interface for technical indicators:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

class IndicatorBase(ComponentBase):
    """Base interface for technical indicators."""
    
    @abstractmethod
    def calculate(self, data: Union[pd.DataFrame, np.ndarray, List[float]]) -> Union[float, np.ndarray, pd.Series]:
        """
        Calculate indicator value(s).
        
        Args:
            data: Price/volume data to calculate indicator from
            
        Returns:
            Calculated indicator values
        """
        pass
    
    @abstractmethod
    def update(self, value: float) -> float:
        """
        Update indicator with a new value.
        
        Args:
            value: New data point
            
        Returns:
            Updated indicator value
        """
        pass
    
    @property
    @abstractmethod
    def value(self) -> float:
        """Get current indicator value."""
        pass
    
    @property
    @abstractmethod
    def ready(self) -> bool:
        """Whether indicator has enough data to provide valid values."""
        pass
```

### RuleBase

Interface for trading rules:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

class RuleBase(ComponentBase):
    """Base interface for trading rules."""
    
    @abstractmethod
    def evaluate(self, data: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Evaluate rule on data.
        
        Args:
            data: Data dictionary with market/indicator values
            
        Returns:
            Tuple of (rule_triggered, signal_strength)
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get rule parameters.
        
        Returns:
            Dict of parameter names to values
        """
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set rule parameters.
        
        Args:
            parameters: Dict of parameter names to values
        """
        pass
    
    @property
    @abstractmethod
    def parameter_space(self) -> Dict[str, List[Any]]:
        """Get rule parameter space for optimization."""
        pass
    
    @property
    @abstractmethod
    def weight(self) -> float:
        """Get rule weight for composite strategies."""
        pass
    
    @weight.setter
    @abstractmethod
    def weight(self, value: float) -> None:
        """Set rule weight."""
        pass
```

## Risk Module Interfaces

### RiskManagerBase

Interface for risk management components:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class RiskManagerBase(ComponentBase):
    """Base interface for risk management components."""
    
    @abstractmethod
    def on_signal(self, event: Dict[str, Any]) -> None:
        """
        Process signal event.
        
        Args:
            event: Signal event data
        """
        pass
    
    @abstractmethod
    def size_position(self, signal: Dict[str, Any]) -> float:
        """
        Determine position size based on signal and risk parameters.
        
        Args:
            signal: Signal dictionary
            
        Returns:
            Position size
        """
        pass
    
    @abstractmethod
    def validate_order(self, order: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate order against risk constraints.
        
        Args:
            order: Order dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def emit_order(self, order_data: Dict[str, Any]) -> bool:
        """
        Emit order event.
        
        Args:
            order_data: Order data dictionary
            
        Returns:
            bool: Whether order was successfully emitted
        """
        pass
    
    @property
    @abstractmethod
    def risk_limits(self) -> Dict[str, Any]:
        """Get active risk limits."""
        pass
    
    @risk_limits.setter
    @abstractmethod
    def risk_limits(self, limits: Dict[str, Any]) -> None:
        """Set risk limits."""
        pass
```

### PortfolioBase

Interface for portfolio tracking components:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime

class PortfolioBase(ComponentBase):
    """Base interface for portfolio tracking components."""
    
    @abstractmethod
    def on_fill(self, event: Dict[str, Any]) -> None:
        """
        Process fill event.
        
        Args:
            event: Fill event data
        """
        pass
    
    @abstractmethod
    def on_bar(self, event: Dict[str, Any]) -> None:
        """
        Process bar event for position marking.
        
        Args:
            event: Bar event data
        """
        pass
    
    @abstractmethod
    def update_positions(self, bar_data: Dict[str, Any]) -> None:
        """
        Update positions based on new market data.
        
        Args:
            bar_data: Bar data dictionary
        """
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get position details for a symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Position dictionary
        """
        pass
    
    @abstractmethod
    def get_portfolio_value(self) -> float:
        """
        Get current portfolio value.
        
        Returns:
            Total portfolio value
        """
        pass
    
    @abstractmethod
    def get_equity_curve(self) -> List[Dict[str, Any]]:
        """
        Get portfolio equity curve history.
        
        Returns:
            List of equity points
        """
        pass
    
    @abstractmethod
    def get_trades(self) -> List[Dict[str, Any]]:
        """
        Get trade history.
        
        Returns:
            List of completed trades
        """
        pass
    
    @property
    @abstractmethod
    def positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all current positions."""
        pass
    
    @property
    @abstractmethod
    def cash(self) -> float:
        """Get available cash."""
        pass
```

### PositionSizerBase

Interface for position sizing components:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class PositionSizerBase(ComponentBase):
    """Base interface for position sizing components."""
    
    @abstractmethod
    def size_position(self, signal: Dict[str, Any], portfolio: 'PortfolioBase') -> float:
        """
        Calculate position size for a signal.
        
        Args:
            signal: Signal dictionary
            portfolio: Portfolio component
            
        Returns:
            Position size
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get position sizer parameters.
        
        Returns:
            Dict of parameter names to values
        """
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set position sizer parameters.
        
        Args:
            parameters: Dict of parameter names to values
        """
        pass
```

### RiskLimitBase

Interface for risk limit components:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

class RiskLimitBase(ComponentBase):
    """Base interface for risk limit components."""
    
    @abstractmethod
    def validate_order(self, order: Dict[str, Any], portfolio: 'PortfolioBase') -> Tuple[bool, Optional[str]]:
        """
        Validate order against risk limit.
        
        Args:
            order: Order dictionary
            portfolio: Portfolio component
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get risk limit parameters.
        
        Returns:
            Dict of parameter names to values
        """
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set risk limit parameters.
        
        Args:
            parameters: Dict of parameter names to values
        """
        pass
```

## Execution Module Interfaces

### BrokerBase

Interface for broker components:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

class BrokerBase(ComponentBase):
    """Base interface for broker components."""
    
    @abstractmethod
    def on_order(self, event: Dict[str, Any]) -> None:
        """
        Process order event.
        
        Args:
            event: Order event data
        """
        pass
    
    @abstractmethod
    def execute_order(self, order_data: Dict[str, Any], price: Optional[float] = None) -> bool:
        """
        Execute an order.
        
        Args:
            order_data: Order data dictionary
            price: Override price (None to use market price)
            
        Returns:
            bool: Whether order was successfully executed
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            bool: Whether order was successfully cancelled
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get status of an order.
        
        Args:
            order_id: ID of order to check
            
        Returns:
            Order status dictionary
        """
        pass
    
    @abstractmethod
    def get_orders(self, symbol: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get orders filtered by symbol and/or status.
        
        Args:
            symbol: Symbol to filter by (None for all)
            status: Status to filter by (None for all)
            
        Returns:
            List of order dictionaries
        """
        pass
```

### SimulatorBase

Interface for market simulation components:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class SimulatorBase(ComponentBase):
    """Base interface for market simulation components."""
    
    @abstractmethod
    def on_bar(self, event: Dict[str, Any]) -> None:
        """
        Process bar event.
        
        Args:
            event: Bar event data
        """
        pass
    
    @abstractmethod
    def check_pending_orders(self, bar_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check pending orders against new bar data.
        
        Args:
            bar_data: Bar data dictionary
            
        Returns:
            List of filled order dictionaries
        """
        pass
    
    @abstractmethod
    def calculate_execution_price(self, order: Dict[str, Any], bar_data: Dict[str, Any]) -> float:
        """
        Calculate execution price for an order.
        
        Args:
            order: Order dictionary
            bar_data: Bar data dictionary
            
        Returns:
            Execution price
        """
        pass
    
    @abstractmethod
    def apply_slippage(self, price: float, direction: str, quantity: float) -> float:
        """
        Apply slippage to execution price.
        
        Args:
            price: Base price
            direction: Order direction
            quantity: Order quantity
            
        Returns:
            Price with slippage applied
        """
        pass
    
    @abstractmethod
    def calculate_commission(self, price: float, quantity: float) -> float:
        """
        Calculate commission for a trade.
        
        Args:
            price: Execution price
            quantity: Order quantity
            
        Returns:
            Commission amount
        """
        pass
```

### OrderManagerBase

Interface for order management components:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class OrderManagerBase(ComponentBase):
    """Base interface for order management components."""
    
    @abstractmethod
    def create_order(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an order from a signal.
        
        Args:
            signal: Signal dictionary
            
        Returns:
            Order dictionary
        """
        pass
    
    @abstractmethod
    def update_order(self, order_id: str, status: str, **kwargs) -> None:
        """
        Update order status.
        
        Args:
            order_id: ID of order to update
            status: New status
            **kwargs: Additional order fields to update
        """
        pass
    
    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an order by ID.
        
        Args:
            order_id: ID of order to get
            
        Returns:
            Order dictionary or None if not found
        """
        pass
    
    @abstractmethod
    def get_orders_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get orders for a symbol.
        
        Args:
            symbol: Symbol to get orders for
            
        Returns:
            List of order dictionaries
        """
        pass
    
    @abstractmethod
    def get_active_orders(self) -> List[Dict[str, Any]]:
        """
        Get all active orders.
        
        Returns:
            List of active order dictionaries
        """
        pass
```

## Core Event System Interfaces

### EventBusBase

Interface for event bus components:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable

class EventBusBase(ComponentBase):
    """Base interface for event bus components."""
    
    @abstractmethod
    def publish(self, event: Dict[str, Any]) -> bool:
        """
        Publish an event.
        
        Args:
            event: Event dictionary
            
        Returns:
            bool: Whether event was successfully published
        """
        pass
    
    @abstractmethod
    def subscribe(self, event_type: str, handler: Callable[[Dict[str, Any]], None], 
                 context: Optional[Any] = None) -> None:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to
            handler: Function to call when events occur
            context: Optional context to associate with the subscription
        """
        pass
    
    @abstractmethod
    def unsubscribe(self, event_type: str, handler: Callable[[Dict[str, Any]], None]) -> bool:
        """
        Unsubscribe from events of a specific type.
        
        Args:
            event_type: Type of events to unsubscribe from
            handler: Handler to unsubscribe
            
        Returns:
            bool: Whether handler was successfully unsubscribed
        """
        pass
    
    @abstractmethod
    def unsubscribe_all(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Unsubscribe a handler from all event types.
        
        Args:
            handler: Handler to unsubscribe
        """
        pass
```

### EventContextBase

Interface for event context components:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class EventContextBase(ABC):
    """Base interface for event context components."""
    
    @abstractmethod
    def __enter__(self) -> 'EventContextBase':
        """
        Enter context and activate it.
        
        Returns:
            Self for context management
        """
        pass
    
    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit context and deactivate it.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get context name."""
        pass
```

## Analytics Module Interfaces

### PerformanceAnalyticsBase

Interface for performance analytics components:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd

class PerformanceAnalyticsBase(ComponentBase):
    """Base interface for performance analytics components."""
    
    @abstractmethod
    def calculate_returns(self, equity_curve: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate return metrics from equity curve.
        
        Args:
            equity_curve: List of equity curve points
            
        Returns:
            Dict of return metrics
        """
        pass
    
    @abstractmethod
    def calculate_risk_metrics(self, equity_curve: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate risk metrics from equity curve.
        
        Args:
            equity_curve: List of equity curve points
            
        Returns:
            Dict of risk metrics
        """
        pass
    
    @abstractmethod
    def calculate_trade_statistics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate trade statistics.
        
        Args:
            trades: List of trade records
            
        Returns:
            Dict of trade statistics
        """
        pass
    
    @abstractmethod
    def generate_report(self, equity_curve: List[Dict[str, Any]], 
                       trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate performance report.
        
        Args:
            equity_curve: List of equity curve points
            trades: List of trade records
            
        Returns:
            Report dictionary with all metrics
        """
        pass
```

### ReportGeneratorBase

Interface for report generation components:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class ReportGeneratorBase(ComponentBase):
    """Base interface for report generation components."""
    
    @abstractmethod
    def generate_text_report(self, results: Dict[str, Any]) -> str:
        """
        Generate text report from results.
        
        Args:
            results: Results dictionary
            
        Returns:
            Text report
        """
        pass
    
    @abstractmethod
    def generate_html_report(self, results: Dict[str, Any]) -> str:
        """
        Generate HTML report from results.
        
        Args:
            results: Results dictionary
            
        Returns:
            HTML report
        """
        pass
    
    @abstractmethod
    def generate_equity_chart(self, equity_curve: List[Dict[str, Any]]) -> Any:
        """
        Generate equity curve chart.
        
        Args:
            equity_curve: List of equity curve points
            
        Returns:
            Chart object
        """
        pass
    
    @abstractmethod
    def save_report(self, report: str, file_path: str) -> bool:
        """
        Save report to a file.
        
        Args:
            report: Report content
            file_path: Path to save to
            
        Returns:
            bool: Whether report was successfully saved
        """
        pass
```

## Dependency Injection Interfaces

### ContainerBase

Interface for dependency injection container:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, Callable

class ContainerBase(ABC):
    """Base interface for dependency injection container."""
    
    @abstractmethod
    def register(self, name: str, component_class: Type, singleton: bool = True) -> None:
        """
        Register a component class.
        
        Args:
            name: Component name
            component_class: Component class
            singleton: Whether to use singleton pattern
        """
        pass
    
    @abstractmethod
    def register_instance(self, name: str, instance: Any) -> None:
        """
        Register a component instance.
        
        Args:
            name: Component name
            instance: Component instance
        """
        pass
    
    @abstractmethod
    def register_factory(self, name: str, factory: Callable[..., Any]) -> None:
        """
        Register a component factory.
        
        Args:
            name: Component name
            factory: Factory function
        """
        pass
    
    @abstractmethod
    def get(self, name: str) -> Any:
        """
        Get a component by name.
        
        Args:
            name: Component name
            
        Returns:
            Component instance
        """
        pass
    
    @abstractmethod
    def has(self, name: str) -> bool:
        """
        Check if a component exists.
        
        Args:
            name: Component name
            
        Returns:
            bool: Whether component exists
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset container state."""
        pass
```

## Implementation Strategy

### 1. Define Interface Hierarchies

Create clear interface hierarchies with proper inheritance:

```
ComponentBase
├── DataHandlerBase
├── StrategyBase
├── RiskManagerBase
├── PortfolioBase
├── BrokerBase
└── PerformanceAnalyticsBase
```

### 2. Implement Base Classes

Implement abstract base classes for each interface:

```python
class Component(ComponentBase):
    """Base implementation of ComponentBase interface."""
    
    def __init__(self, name, parameters=None):
        self.name = name
        self.parameters = parameters or {}
        self.initialized = False
        self.running = False
        
    def initialize(self, context):
        # Extract common dependencies
        self.event_bus = context.get('event_bus')
        self.logger = context.get('logger')
        self.config = context.get('config')
        
        # Set initialized flag
        self.initialized = True
        
    def start(self):
        if not self.initialized:
            raise RuntimeError("Component must be initialized before starting")
        self.running = True
        
    def stop(self):
        self.running = False
        
    def reset(self):
        # Reset state but maintain configuration
        pass
        
    def teardown(self):
        # Unsubscribe from events
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.unsubscribe_all(self)
            
        # Reset flags
        self.initialized = False
        self.running = False
```

### 3. Update Existing Components

Update existing components to implement the interfaces:

```python
class HistoricalDataHandler(DataHandlerBase):
    """Historical data handler implementation."""
    
    def __init__(self, name="historical_data_handler", parameters=None):
        super().__init__(name, parameters)
        self.data = {}
        self.current_index = {}
        self.bars = {}
        
    def load_data(self, symbols):
        # Implementation
        pass
    
    def update_bars(self):
        # Implementation
        pass
    
    def get_latest_bar(self, symbol):
        # Implementation
        pass
    
    # Additional method implementations
```

### 4. Create Factory Pattern

Implement factories for each component type:

```python
class StrategyFactory:
    """Factory for creating strategy instances."""
    
    @staticmethod
    def create(strategy_type, name, parameters=None):
        """
        Create a strategy instance.
        
        Args:
            strategy_type: Type of strategy to create
            name: Strategy name
            parameters: Strategy parameters
            
        Returns:
            Strategy instance
        """
        if strategy_type == "ma_crossover":
            return MovingAverageCrossoverStrategy(name, parameters)
        elif strategy_type == "breakout":
            return BreakoutStrategy(name, parameters)
        elif strategy_type == "mean_reversion":
            return MeanReversionStrategy(name, parameters)
        elif strategy_type == "composite":
            return CompositeStrategy(name, parameters)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
```

### 5. Update Configuration System

Enhance configuration system to support interface-based components:

```yaml
# In configuration.yaml
components:
  data_handler:
    type: historical
    parameters:
      data_dir: /path/to/data
      
  strategy:
    type: ma_crossover
    parameters:
      fast_window: 10
      slow_window: 30
      
  risk_manager:
    type: standard
    parameters:
      position_size_pct: 0.05
      max_positions: 10
      
  broker:
    type: simulated
    parameters:
      slippage_model: fixed
      slippage_pct: 0.001
```

### 6. Update Bootstrap Process

Enhance bootstrap process to use the interfaces:

```python
def bootstrap(config_file):
    """Bootstrap the system with interface-based components."""
    # Load configuration
    config = Config()
    config.load(config_file)
    
    # Create container
    container = Container()
    
    # Register core components
    container.register_instance('config', config)
    container.register('event_bus', EventBus)
    container.register('logger', Logger)
    
    # Register factories
    container.register_factory('data_handler', lambda: create_data_handler(config))
    container.register_factory('strategy', lambda: create_strategy(config))
    container.register_factory('risk_manager', lambda: create_risk_manager(config))
    container.register_factory('broker', lambda: create_broker(config))
    
    # Create and initialize components
    event_bus = container.get('event_bus')
    data_handler = container.get('data_handler')
    strategy = container.get('strategy')
    risk_manager = container.get('risk_manager')
    broker = container.get('broker')
    
    # Create context
    context = {
        'event_bus': event_bus,
        'logger': container.get('logger'),
        'config': config
    }
    
    # Initialize components
    data_handler.initialize(context)
    strategy.initialize(context)
    risk_manager.initialize(context)
    broker.initialize(context)
    
    return container, config
```

## Interface Benefits

Implementing these interfaces provides several key benefits:

1. **Clear Boundaries**: Components interact through well-defined interfaces, not implementation details
2. **Testability**: Components can be easily mocked and tested in isolation
3. **Replaceability**: Implementations can be swapped without affecting other components
4. **Documentation**: Interfaces provide clear self-documentation of component responsibilities
5. **Contract Enforcement**: Type hints and abstract methods enforce proper implementation
6. **Dependency Injection**: Components receive dependencies through standardized mechanism

## Example Code

### Example: Strategy Interface Usage

```python
class MovingAverageCrossoverStrategy(StrategyBase):
    """Moving average crossover strategy implementation."""
    
    def __init__(self, name="ma_crossover", parameters=None):
        super().__init__(name, parameters or {})
        # Initialize default parameters
        self.fast_window = self.parameters.get('fast_window', 10)
        self.slow_window = self.parameters.get('slow_window', 30)
        self.position_size = self.parameters.get('position_size', 100)
        
        # Initialize state containers
        self.prices = {}
        self.fast_ma = {}
        self.slow_ma = {}
        self.current_position = {}
        self._parameter_space = {
            'fast_window': list(range(5, 21)),
            'slow_window': list(range(20, 51, 5)),
            'position_size': [50, 100, 200, 500]
        }
        
    def on_bar(self, event):
        # Extract bar data
        bar_data = event.get('data', {})
        
        # Update indicators
        self._update_indicators(bar_data)
        
        # Calculate and emit signals
        signals = self.calculate_signals(bar_data)
        for signal in signals:
            self.emit_signal(**signal)
            
    def calculate_signals(self, bar_data):
        symbol = bar_data.get('symbol')
        signals = []
        
        # Skip if we don't have both MAs
        if symbol not in self.fast_ma or symbol not in self.slow_ma:
            return signals
            
        # Skip if we don't have previous values
        if 'prev_fast_ma' not in self.indicators.get(symbol, {}):
            # Store current values for next bar
            if symbol not in self.indicators:
                self.indicators[symbol] = {}
            self.indicators[symbol]['prev_fast_ma'] = self.fast_ma[symbol]
            self.indicators[symbol]['prev_slow_ma'] = self.slow_ma[symbol]
            return signals
            
        # Get current values
        fast_ma = self.fast_ma[symbol]
        slow_ma = self.slow_ma[symbol]
        
        # Get previous values
        prev_fast_ma = self.indicators[symbol]['prev_fast_ma']
        prev_slow_ma = self.indicators[symbol]['prev_slow_ma']
        
        # Store current values for next bar
        self.indicators[symbol]['prev_fast_ma'] = fast_ma
        self.indicators[symbol]['prev_slow_ma'] = slow_ma
        
        # Get current position
        current_position = self.current_position.get(symbol, 0)
        
        # Check for crossover
        if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
            # Buy signal
            if current_position <= 0:
                signals.append({
                    'symbol': symbol,
                    'direction': 'BUY',
                    'quantity': self.position_size,
                    'price': bar_data['close'],
                    'reason': 'ma_crossover_up'
                })
                self.current_position[symbol] = self.position_size
        elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma:
            # Sell signal
            if current_position >= 0:
                signals.append({
                    'symbol': symbol,
                    'direction': 'SELL',
                    'quantity': self.position_size,
                    'price': bar_data['close'],
                    'reason': 'ma_crossover_down'
                })
                self.current_position[symbol] = -self.position_size
                
        return signals
        
    def emit_signal(self, symbol, direction, quantity, price, **kwargs):
        if not hasattr(self, 'event_bus') or not self.event_bus:
            return False
            
        # Create signal event
        signal_data = {
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'price': price,
            'timestamp': kwargs.get('timestamp', datetime.now()),
            'strategy': self.name,
            'reason': kwargs.get('reason', 'unknown')
        }
        
        # Add any additional data
        for k, v in kwargs.items():
            if k not in signal_data:
                signal_data[k] = v
                
        # Create and emit event
        event = {
            'type': 'SIGNAL',
            'data': signal_data
        }
        
        return self.event_bus.publish(event)
        
    def _update_indicators(self, bar_data):
        symbol = bar_data.get('symbol')
        if not symbol:
            return
            
        close = bar_data.get('close')
        if close is None:
            return
            
        # Initialize price history if needed
        if symbol not in self.prices:
            self.prices[symbol] = []
            
        # Add price to history
        self.prices[symbol].append(close)
        
        # Limit history length
        max_window = max(self.fast_window, self.slow_window)
        if len(self.prices[symbol]) > max_window + 10:  # Keep a few extra for efficiency
            self.prices[symbol] = self.prices[symbol][-max_window-10:]
            
        # Calculate MAs if we have enough data
        if len(self.prices[symbol]) >= self.slow_window:
            self.fast_ma[symbol] = sum(self.prices[symbol][-self.fast_window:]) / self.fast_window
            self.slow_ma[symbol] = sum(self.prices[symbol][-self.slow_window:]) / self.slow_window
            
    def get_parameters(self):
        return {
            'fast_window': self.fast_window,
            'slow_window': self.slow_window,
            'position_size': self.position_size
        }
        
    def set_parameters(self, parameters):
        if 'fast_window' in parameters:
            self.fast_window = parameters['fast_window']
        if 'slow_window' in parameters:
            self.slow_window = parameters['slow_window']
        if 'position_size' in parameters:
            self.position_size = parameters['position_size']
            
        # Clear cached calculations
        self.prices = {}
        self.fast_ma = {}
        self.slow_ma = {}
        self.indicators = {}
        
    def reset(self):
        super().reset()
        # Clear state
        self.prices = {}
        self.fast_ma = {}
        self.slow_ma = {}
        self.current_position = {}
        self.indicators = {}
        
    @property
    def parameter_space(self):
        return self._parameter_space
```

## Conclusion

This interface-based design establishes clear boundaries between modules in the ADMF-Trader system. By defining explicit contracts through interfaces, we ensure proper separation of concerns, enabling more robust testing, easier extension, and better maintainability. 

The interfaces also facilitate dependency injection, allowing components to be configured and composed in different ways without tight coupling. This approach enables the system to be more adaptable to different use cases and environments.

As development progresses, these interfaces will serve as the foundation for implementing concrete components while maintaining architectural integrity.