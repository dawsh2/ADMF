#!/usr/bin/env python3
"""
Fix for the portfolio cash accounting bug.

The bug occurs when a position flips (e.g., from long to short). 
The current code updates cash for the entire fill quantity, 
but it should only update cash for the net new position after closing the existing one.

Example:
- Have 100 long shares
- Sell 200 shares
- First 100 close the long (cash already accounted for when opening the long)
- Next 100 open a short (only these 100 should affect cash)
- Current bug: Updates cash for all 200 shares
"""

def create_fixed_on_fill_method():
    """
    Returns the corrected on_fill method code that properly handles cash updates
    when positions flip or are partially closed.
    """
    
    return '''
    def on_fill(self, event: Event):
        self.logger.debug(f"{self.name} received FILL event")
        fill_data = event.payload
        fill_id = fill_data.get('fill_id', 'NO_ID')
        self.logger.debug(f"Processing FILL {fill_id} - current trades: {len(self._trade_log)}")
        if not (isinstance(fill_data, dict) and all(k in fill_data for k in ['symbol', 'timestamp', 'quantity_filled', 'fill_price', 'direction'])):
            self.logger.error(f"'{self.name}' received incomplete or invalid FILL data: {fill_data}"); return

        symbol = fill_data['symbol']
        timestamp = fill_data['timestamp']
        quantity_filled = float(fill_data['quantity_filled'])
        fill_price = float(fill_data['fill_price'])
        commission = float(fill_data.get('commission', 0.0))
        fill_direction_is_buy = fill_data['direction'].upper() == 'BUY'

        # Record the pre-fill state for validation
        pre_fill_cash = self.current_cash
        
        self.current_cash -= commission
        self.logger.debug(f"Commission deducted: {commission:.2f}, Cash before fill: {pre_fill_cash:.2f}, Cash after commission: {self.current_cash:.2f}")
        
        # Log trade for debugging train/test performance
        side = "BUY" if fill_direction_is_buy else "SELL"
        self.logger.info(f"TRADE: {side} {quantity_filled} {symbol} @ ${fill_price:.2f}, Portfolio Value: ${self.current_total_value:.2f}")
        
        active_regime = self._get_current_regime()

        pos = self.open_positions.get(symbol)
        
        # Initialize new position if none exists for the symbol
        if not pos:
            pos = {
                'quantity': 0.0, 'avg_entry_price': 0.0, 'cost_basis': 0.0,
                'entry_timestamp': timestamp, 'trade_id': str(uuid.uuid4()),
                'current_segment_entry_price': fill_price,
                'current_segment_regime': active_regime,
                'initial_entry_regime_for_trade': active_regime
            }
            self.open_positions[symbol] = pos
            self.logger.info(f"New position structure initialized for {symbol} due to fill.")

        current_pos_qty = pos['quantity']
        
        # Determine P&L if fill reduces or closes an existing position
        pnl_from_this_fill = 0.0
        qty_closed = 0.0

        if (current_pos_qty > 0 and not fill_direction_is_buy) or \\
           (current_pos_qty < 0 and fill_direction_is_buy): # Closing/reducing existing position
            
            qty_closed = min(abs(current_pos_qty), quantity_filled)
            segment_entry_price = pos['current_segment_entry_price']

            if current_pos_qty > 0: # Closing/reducing long
                pnl_from_this_fill = (fill_price - segment_entry_price) * qty_closed
                self.logger.info(f"Closed LONG segment {qty_closed:.2f} {symbol} at {fill_price:.2f} (entry {segment_entry_price:.2f}). PnL: {pnl_from_this_fill:.2f} in regime '{pos['current_segment_regime']}'.")
            else: # Covering/reducing short
                pnl_from_this_fill = (segment_entry_price - fill_price) * qty_closed
                self.logger.info(f"Covered SHORT segment {qty_closed:.2f} {symbol} at {fill_price:.2f} (entry {segment_entry_price:.2f}). PnL: {pnl_from_this_fill:.2f} in regime '{pos['current_segment_regime']}'.")
            
            self.realized_pnl += pnl_from_this_fill
            # Track both entry and exit regimes for boundary trades
            entry_regime = pos['current_segment_regime']
            exit_regime = self._get_current_regime()
            
            # Create a unique identifier for this trade segment
            segment_id = str(uuid.uuid4())
            
            # Create more detailed trade log entry
            trade_entry = {
                'symbol': symbol,
                'trade_id': pos['trade_id'],
                'segment_id': segment_id,
                'entry_timestamp': pos['entry_timestamp'],
                'exit_timestamp': timestamp,
                'direction': 'LONG' if current_pos_qty > 0 else 'SHORT',
                'entry_price': segment_entry_price,
                'exit_price': fill_price,
                'quantity': qty_closed,
                'commission': commission,
                'pnl': pnl_from_this_fill,
                'entry_regime': entry_regime,
                'exit_regime': exit_regime,
                'is_boundary_trade': (entry_regime != exit_regime),
                # For backward compatibility, keep the 'regime' field as the entry regime
                'regime': entry_regime
            }
            
            self._trade_log.append(trade_entry)
            if len(self._trade_log) % 10 == 0:  # Log every 10 trades
                self.logger.info(f"Trade #{len(self._trade_log)} completed")
            
            # Log boundary trades specifically for analysis
            if entry_regime != exit_regime:
                self.logger.info(
                    f"Boundary trade detected: {trade_entry['direction']} {trade_entry['quantity']:.2f} {symbol} " +
                    f"opened in '{entry_regime}' and closed in '{exit_regime}'. " +
                    f"PnL: {pnl_from_this_fill:.2f}, Segment ID: {segment_id}"
                )

        # CRITICAL FIX: Calculate the quantity that actually affects cash
        # This is the quantity that opens or adds to a position, NOT the quantity used to close
        qty_affecting_cash = quantity_filled - qty_closed
        
        # Update position quantity and cash
        pre_update_cash = self.current_cash
        pre_update_quantity = pos['quantity']
        
        if fill_direction_is_buy:
            pos['quantity'] += quantity_filled
            # Only update cash for the portion that's not closing an existing short
            self.current_cash -= qty_affecting_cash * fill_price
            self.logger.debug(f"BUY - Cash impact: -{qty_affecting_cash * fill_price:.2f} (on {qty_affecting_cash} shares, {qty_closed} shares closed existing), Cash before: {pre_update_cash:.2f}, Cash after: {self.current_cash:.2f}")
            self.logger.debug(f"BUY - Quantity impact: +{quantity_filled:.2f}, Quantity before: {pre_update_quantity:.2f}, Quantity after: {pos['quantity']:.2f}")
        else: # SELL
            pos['quantity'] -= quantity_filled
            # Only update cash for the portion that's not closing an existing long
            self.current_cash += qty_affecting_cash * fill_price
            self.logger.debug(f"SELL - Cash impact: +{qty_affecting_cash * fill_price:.2f} (on {qty_affecting_cash} shares, {qty_closed} shares closed existing), Cash before: {pre_update_cash:.2f}, Cash after: {self.current_cash:.2f}")
            self.logger.debug(f"SELL - Quantity impact: -{quantity_filled:.2f}, Quantity before: {pre_update_quantity:.2f}, Quantity after: {pos['quantity']:.2f}")

        # Handle position state changes (new, flipped, modified average price)
        if abs(current_pos_qty) < 1e-9 and abs(pos['quantity']) > 1e-9 : # Opened new position (was flat)
            pos['trade_id'] = str(uuid.uuid4()) # New trade ID for this new holding period
            pos['entry_timestamp'] = timestamp
            pos['avg_entry_price'] = fill_price
            pos['cost_basis'] = pos['quantity'] * fill_price # Signed quantity
            pos['current_segment_entry_price'] = fill_price
            pos['current_segment_regime'] = active_regime
            pos['initial_entry_regime_for_trade'] = active_regime
            direction_str = "LONG" if pos['quantity'] > 0 else "SHORT"
            self.logger.info(f"Opened {direction_str} {abs(pos['quantity']):.2f} {symbol} at {fill_price:.2f} in regime '{active_regime}'. Trade ID: {pos['trade_id']}")
        
        elif (current_pos_qty > 0 and pos['quantity'] < -1e-9) or \\
             (current_pos_qty < 0 and pos['quantity'] > 1e-9): # Flipped position
            # A position has flipped from long to short or vice versa
            pos['trade_id'] = str(uuid.uuid4()) # New trade ID for the new direction
            pos['entry_timestamp'] = timestamp
            
            # Log the flip details for debugging
            old_direction = "LONG" if current_pos_qty > 0 else "SHORT"
            new_direction = "SHORT" if current_pos_qty > 0 else "LONG"
            self.logger.debug(
                f"Position flip from {old_direction} {abs(current_pos_qty):.2f} to {new_direction} {abs(pos['quantity']):.2f} for {symbol}. "
                f"Fill price: {fill_price:.2f}, Old avg price: {pos['avg_entry_price']:.2f}"
            )
            
            # The flip price is the new basis - this is correct for the new position
            pos['avg_entry_price'] = fill_price 
            
            # When a position flips, the cost basis calculation needs special handling
            # For the new position, cost basis is just the flip price × new quantity
            pos['cost_basis'] = pos['quantity'] * fill_price # Signed quantity
            
            # Reset segment tracking for the new position
            pos['current_segment_entry_price'] = fill_price
            pos['current_segment_regime'] = active_regime
            pos['initial_entry_regime_for_trade'] = active_regime
            
            direction_str = "LONG" if pos['quantity'] > 0 else "SHORT"
            self.logger.info(f"Flipped to {direction_str} {abs(pos['quantity']):.2f} {symbol} at {fill_price:.2f} in regime '{active_regime}'. New Trade ID: {pos['trade_id']}")

        elif abs(pos['quantity']) > 1e-9 : # Added to existing position (or reduced without flipping/closing)
            if qty_closed == 0: # Only adding to position, not closing any part
                if pos['current_segment_regime'] == active_regime:
                    # Different logic for average price calculation depending on position type (long vs short)
                    new_total_abs_qty = abs(current_pos_qty) + quantity_filled
                    
                    if current_pos_qty > 0 and fill_direction_is_buy:  # Adding to long position
                        # Weighted average for longs is straightforward
                        pos['avg_entry_price'] = ((current_pos_qty * pos['avg_entry_price']) + (quantity_filled * fill_price)) / (current_pos_qty + quantity_filled)
                        self.logger.debug(f"Added to LONG {symbol}. Calc: ({current_pos_qty} * {pos['avg_entry_price']}) + ({quantity_filled} * {fill_price}) / {current_pos_qty + quantity_filled}")
                    
                    elif current_pos_qty < 0 and not fill_direction_is_buy:  # Adding to short position
                        # For shorts, we average the prices at which we borrowed/sold shares
                        # The sign is important here - short quantity is negative
                        total_value_borrowed = (abs(current_pos_qty) * pos['avg_entry_price']) + (quantity_filled * fill_price)
                        pos['avg_entry_price'] = total_value_borrowed / (abs(current_pos_qty) + quantity_filled)
                        self.logger.debug(f"Added to SHORT {symbol}. Calc: ({abs(current_pos_qty)} * {pos['avg_entry_price']}) + ({quantity_filled} * {fill_price}) / {abs(current_pos_qty) + quantity_filled}")
                    
                    pos['current_segment_entry_price'] = pos['avg_entry_price'] # Assuming segment avg price tracks overall avg if not flipping regime
                else: # Regime changed while adding to position
                    pos['current_segment_entry_price'] = fill_price
                    pos['current_segment_regime'] = active_regime
                
                pos['cost_basis'] = pos['quantity'] * pos['avg_entry_price'] # Update cost basis based on new average
                direction_str = "LONG" if pos['quantity'] > 0 else "SHORT"
                self.logger.info(f"Added to {direction_str} {symbol}. Old Qty: {current_pos_qty:.2f}, New Qty: {pos['quantity']:.2f}, New Avg Price: {pos['avg_entry_price']:.2f}")
            # If qty_closed > 0, it means part of the position was closed, and if pos['quantity'] is still non-zero,
            # it means the fill was larger than the existing position (a flip), which is handled above.
            # If fill was smaller than existing pos, it's a partial close, P&L logged, quantity reduced.
            # avg_entry_price of remaining position does not change on partial close.
            # current_segment_entry_price also does not change unless regime changes.
            if pos['current_segment_regime'] != active_regime:
                 self.logger.info(f"Regime for open position {symbol} changed from {pos['current_segment_regime']} to {active_regime} during fill.")
                 pos['current_segment_entry_price'] = fill_price # Start new segment at current fill price
                 pos['current_segment_regime'] = active_regime


        if abs(pos['quantity']) < 1e-9: # Position is now flat
            self.logger.info(f"Position in {symbol} fully closed and flattened after fill processing.")
            del self.open_positions[symbol]
        
        self._update_portfolio_value(timestamp)
'''

if __name__ == "__main__":
    print("Portfolio Cash Accounting Bug Fix")
    print("=" * 50)
    print("\nThe bug is in the on_fill method of BasicPortfolio:")
    print("- When a position flips (e.g., long 100 -> short 100 via a sell 200)")
    print("- The current code updates cash for the ENTIRE fill quantity (200)")
    print("- But it should only update cash for the net new position (100)")
    print("\nThe fix calculates qty_affecting_cash = quantity_filled - qty_closed")
    print("This ensures cash is only updated for the portion that opens/adds to positions")
    print("\nTo apply this fix, the on_fill method in basic_portfolio.py needs to be updated.")