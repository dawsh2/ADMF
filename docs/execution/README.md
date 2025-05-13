# Execution Module

The Execution module handles order processing, market simulation and backtest coordination. Its primary responsibility is to process ORDER events from the Risk module and generate FILL events.

> **Important Architectural Note**: The Execution module ONLY processes ORDER events from the Risk module. It does NOT interact directly with SIGNAL events, which are handled exclusively by the Risk module.

## Key Components & Standard Implementation

For detailed implementation specifications, see the [Implementation Guide](IMPLEMENTATION.md).

## Broker

```
BrokerBase (Abstract)
  └── initialize(context)
  └── on_order(order_event)
  └── execute_order(order_data, price=None)
  └── cancel_order(order_id)
  └── get_order_status(order_id)
```

The Broker processes orders received from the Order Manager and generates fill events when orders are executed. It implements slippage and commission models for realistic execution simulation.

## Order Manager

```
OrderManager
  └── initialize(context)
  └── on_order(order_event)      // Process orders from Risk module
  └── on_fill(fill_event)        // Process fills from broker
  └── create_order(order_data)   // Create orders (for internal use)
  └── cancel_order(order_id)     // Cancel orders
  └── get_order(order_id)        // Get order data
```

The Order Manager receives ORDER events from the Risk module, validates them, and forwards them to the appropriate broker. It tracks the full order lifecycle and maintains the system's order state.

> **Note**: The Order Manager does NOT process SIGNAL events directly. All signals are processed by the Risk module.

## Simulated Broker

```python
class SimulatedBroker(BrokerBase):
    def on_bar(bar_event)                          # Process market data updates
    def _process_pending_orders(symbol)            # Process queued orders
    def _process_active_orders(symbol)             # Check conditions for limit/stop orders
    def _execute_market_order(order_data)          # Execute market orders
    def _apply_slippage(price, quantity)           # Apply realistic slippage
    def _calculate_commission(quantity, price)     # Calculate trade commission
```

The SimulatedBroker implements realistic market simulation with configurable slippage and commission models.

## Backtest Coordinator

```
BacktestCoordinator
  └── initialize(context)         # Setup dependencies
  └── setup()                     # Initialize all components
  └── run()                       # Run the entire backtest
  └── _close_positions()          # Close open positions at end of test
  └── _collect_results()          # Gather performance metrics
  └── _calculate_statistics()     # Calculate performance statistics
```

The Backtest Coordinator orchestrates the entire backtesting process by managing the component lifecycle and collecting results.

## Implementation Considerations

For detailed guidance on specific implementation challenges in the Execution module, see the [Implementation Guide](IMPLEMENTATION.md), which covers:

- Responsibility boundaries with the Risk module
- Order management standardization
- State reset protocol
- Event flow patterns
- Tricky implementation areas (order deduplication, PnL calculation, etc.)
- Passthrough functionality for testing strategies without execution effects