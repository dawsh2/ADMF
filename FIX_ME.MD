# FIX_ME.MD - Minor Bugs and Issues

## P&L Calculation Discrepancy
**Date Found**: 2025-05-25
**Severity**: Low
**Description**: 
- Trade execution prices shown in logs don't match P&L calculation entry prices
- Example: Trade shows "BUY @ $521.12" but P&L calculation uses "entry 521.17"
- Results in ~$5 discrepancy in final portfolio value vs realized P&L
- Final Portfolio: $100,021.49, Initial: $100,000, Gain: $21.49
- But Realized P&L only shows $16.00

**Likely Cause**: 
- Slippage or commission accounting inconsistency
- Display price vs actual fill price mismatch

**Files to Check**:
- `src/execution/simulated_execution_handler.py`
- `src/risk/basic_portfolio.py`

---

## Add new issues below this line