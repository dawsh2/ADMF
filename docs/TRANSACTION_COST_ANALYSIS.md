# Transaction Cost and Slippage Analysis

## Current Implementation

### 1. Commission Model
The system currently uses a **fixed commission per trade** model:

```yaml
# From config.yaml
simulated_execution_handler:
  commission_per_trade: 0.005
  commission_type: "per_share"  # Note: This seems to be ignored in code
```

However, looking at the `SimulatedExecutionHandler` implementation:
```python
# Line 144 in simulated_execution_handler.py
"commission": self._commission_per_trade,  # Use the configured commission
```

**Issues:**
- The `commission_type: "per_share"` in config is not actually implemented
- The commission is applied as a fixed amount per trade, not per share
- Default is 0.0 if not configured (line 25: `self._commission_per_trade: float = 0.0`)

### 2. Slippage Model
Currently, there is **NO slippage modeling**:

```python
# Line 134 in simulated_execution_handler.py
# For MVP, assume immediate fill at the specified price (no slippage beyond what's implied by fill_price)
```

Orders are filled at exactly the price specified in the order:
- Market orders use `simulated_fill_price` from the order
- The risk manager creates orders with `simulated_fill_price` = current bar price
- No bid-ask spread modeling
- No market impact modeling
- No price improvement or adverse selection

### 3. Fill Process
1. Strategy generates signal
2. Risk manager creates order with current bar's price as `simulated_fill_price`
3. Execution handler fills at exactly that price
4. Portfolio deducts fixed commission from cash

## What's Missing

### 1. Realistic Commission Structure
- **Per-share commission**: $0.005/share is common for retail
- **Percentage-based**: 0.1% of trade value for some brokers
- **Tiered pricing**: Volume-based discounts
- **Minimum commission**: Many brokers have a $1 minimum

### 2. Slippage Components
- **Bid-Ask Spread**: Typically 0.01-0.05% for liquid stocks
- **Market Impact**: Proportional to order size
- **Timing Delay**: Price movement between signal and execution
- **Adverse Selection**: Worse fills during volatile periods

### 3. Implementation Recommendations

#### Quick Fix (Minimal Changes)
Add simple slippage to execution handler:
```python
# In simulated_execution_handler.py
slippage_bps = self.get_specific_config("slippage_bps", 5)  # 5 basis points
slippage_factor = slippage_bps / 10000
if direction == "BUY":
    fill_price = fill_price * (1 + slippage_factor)
else:
    fill_price = fill_price * (1 - slippage_factor)
```

#### Proper Implementation
1. **Fix commission calculation**:
   - Actually use the `commission_type` config
   - Calculate per-share commission correctly
   
2. **Add configurable slippage model**:
   - Fixed spread (basis points)
   - Volume-dependent impact
   - Volatility-adjusted spread
   
3. **Make it regime-aware**:
   - Higher spreads in volatile regimes
   - Lower liquidity in trending markets

## Configuration Example
```yaml
execution_handler:
  commission:
    type: "per_share"  # or "per_trade", "percentage"
    rate: 0.005        # $0.005 per share
    minimum: 1.0       # $1 minimum
  slippage:
    base_spread_bps: 5      # 5 basis points base spread
    impact_bps_per_lot: 1   # 1 bp per 100 shares
    volatility_multiplier: 2.0  # Double spread in volatile regimes
```

## Impact on Current Results
With the current implementation:
- **Commission**: $0.005 per trade (essentially free for 100-share trades)
- **Slippage**: 0% (perfect fills at signal price)
- **Total cost**: ~0.005% per round trip (negligible)

This explains why your strategies can be profitable with small edges.
In reality, with proper costs:
- **Commission**: $1 minimum or $0.50 per 100 shares
- **Slippage**: 5-10 bps per side
- **Total cost**: 10-20 bps per round trip (0.1-0.2%)

This would significantly impact strategies with small average gains per trade.