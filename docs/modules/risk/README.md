# Risk Module

The Risk module manages position sizing, risk control, and portfolio tracking. It receives trading signals from the Strategy module, applies risk management rules, and emits properly sized orders.

## Key Components

For detailed implementation specifications, see the [Implementation Guide](IMPLEMENTATION.md).

## Risk Manager

```
RiskManagerBase (Abstract)
  └── initialize(context)
  └── initialize_event_subscriptions()
  └── on_signal(event)
  └── size_position(signal)
  └── validate_order(order)
  └── emit_order(order_data)
  └── reset()
```

The Risk Manager converts signals to orders with appropriate risk controls and position sizing. It enforces risk limits and prevents trades that would violate risk constraints.

## Portfolio Manager

```
PortfolioBase (Abstract)
  └── initialize(context)
  └── initialize_event_subscriptions()
  └── on_fill(event)
  └── on_bar(event)
  └── update_positions(bar)
  └── calculate_equity()
  └── get_position(symbol)
  └── get_portfolio_value()
  └── reset()
```

The Portfolio Manager is the single source of truth for positions and equity. It tracks positions, cash, and calculates portfolio statistics including P&L.

## Position Sizing Strategies

```
PositionSizerBase (Abstract)
  └── calculate_position_size(signal, portfolio, current_position)
  └── reset()

Implementations:
  └── FixedSizer
  └── PercentEquitySizer
  └── PercentRiskSizer
  └── KellySizer
  └── VolatilitySizer
```

Position Sizers calculate appropriate trade sizes based on the portfolio state, signal parameters, and risk configuration.

## Risk Limits

```
RiskLimitBase (Abstract)
  └── check(signal, quantity, portfolio)
  └── reset()

Implementations:
  └── MaxPositionSizeLimit
  └── MaxExposureLimit
  └── MaxDrawdownLimit
  └── MaxLossLimit
  └── MaxPositionsLimit
```

Risk Limits enforce trading constraints and can reject trades that would exceed risk thresholds.

## Position Model

```python
class Position:
    def __init__(self, symbol)
    def update(quantity, price, commission=0.0)
    def mark_to_market(price)
    def calculate_pnl()
    def close(price)
    def get_info()
```

The Position class represents individual security positions with quantity, cost basis, and market value information.

## Implementation Notes

- The Risk module is the sole authority for positions and P&L in the system
- All components provide sensible defaults for configuration parameters
- Risk management can be configured as a passthrough for debugging strategy performance
- Thread safety is implemented for all shared state

For detailed implementation guidance, see the [Implementation Guide](IMPLEMENTATION.md).