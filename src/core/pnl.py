"""
P&L calculation module.

Implements position updates and P&L calculations with Numba optimization.
"""

import numba
from typing import Tuple


@numba.jit(nopython=True, cache=True)
def calculate_position_pnl(
    position_quantity: float,
    avg_entry_price: float,
    current_price: float,
    contract_multiplier: float
) -> Tuple[float, float]:
    """
    Calculate realized and unrealized P&L for a position.

    Args:
        position_quantity: Current position (signed: + long, - short)
        avg_entry_price: Average entry price
        current_price: Current market price
        contract_multiplier: Contract multiplier

    Returns:
        (realized_pnl, unrealized_pnl)

    Note:
        Realized P&L is passed in separately (accumulated from closed trades)
        This function only calculates unrealized P&L
    """
    realized_pnl = 0.0  # Maintained separately

    if position_quantity == 0:
        unrealized_pnl = 0.0
    else:
        # Unrealized P&L = (current_price - avg_entry) * quantity * multiplier
        price_diff = current_price - avg_entry_price
        unrealized_pnl = price_diff * position_quantity * contract_multiplier

    return realized_pnl, unrealized_pnl


@numba.jit(nopython=True, cache=True)
def update_position(
    current_quantity: float,
    current_avg_entry: float,
    fill_quantity: float,
    fill_price: float,
    fill_side: int,
    contract_multiplier: float
) -> Tuple[float, float, float]:
    """
    Update position after a fill.

    Args:
        current_quantity: Current position quantity
        current_avg_entry: Current average entry price
        fill_quantity: Fill quantity (always positive)
        fill_price: Fill price
        fill_side: Fill side (1=buy, -1=sell)
        contract_multiplier: Contract multiplier

    Returns:
        (new_quantity, new_avg_entry, realized_pnl)

    Logic:
        - If fill increases position (same direction): update avg entry
        - If fill reduces position (opposite direction): realize P&L
        - If fill flips position: realize P&L for closed portion, new entry for remainder
    """
    # Signed fill quantity
    signed_fill = fill_quantity * fill_side

    # Check if increasing or decreasing position
    if current_quantity == 0:
        # Opening new position
        new_quantity = signed_fill
        new_avg_entry = fill_price
        realized_pnl = 0.0

    elif (current_quantity > 0 and fill_side > 0) or (current_quantity < 0 and fill_side < 0):
        # Increasing position (same direction)
        total_cost = current_quantity * current_avg_entry + signed_fill * fill_price
        new_quantity = current_quantity + signed_fill
        new_avg_entry = total_cost / new_quantity
        realized_pnl = 0.0

    else:
        # Reducing or flipping position (opposite direction)
        closing_quantity = min(abs(fill_quantity), abs(current_quantity))

        # Calculate realized P&L for closed portion
        if current_quantity > 0:
            # Closing long position
            pnl_per_unit = (fill_price - current_avg_entry) * contract_multiplier
        else:
            # Closing short position
            pnl_per_unit = (current_avg_entry - fill_price) * contract_multiplier

        realized_pnl = pnl_per_unit * closing_quantity

        # Update position
        new_quantity = current_quantity + signed_fill

        if new_quantity == 0:
            # Position fully closed
            new_avg_entry = 0.0
        elif (current_quantity > 0 and new_quantity < 0) or (current_quantity < 0 and new_quantity > 0):
            # Position flipped - new entry price for remaining
            new_avg_entry = fill_price
        else:
            # Position reduced but not flipped - keep entry price
            new_avg_entry = current_avg_entry

    return new_quantity, new_avg_entry, realized_pnl


@numba.jit(nopython=True, cache=True)
def calculate_fill_cost(
    notional_value: float,
    fee_rate: float,
    is_maker: bool
) -> float:
    """
    Calculate transaction fee for a fill.

    Args:
        notional_value: Fill notional (quantity * price * multiplier)
        fee_rate: Fee rate (e.g., 0.0001 = 1 basis point)
        is_maker: True if maker fill (may have rebate)

    Returns:
        Fee amount (positive = cost, negative = rebate)
    """
    fee = notional_value * fee_rate
    return fee


def update_position_with_fill(
    current_quantity: float,
    current_avg_entry: float,
    current_realized_pnl: float,
    fill_quantity: float,
    fill_price: float,
    fill_side: int,
    fee: float,
    contract_multiplier: float = 1.0
) -> Tuple[float, float, float]:
    """
    Update position with fill and fee (non-JIT wrapper).

    Args:
        current_quantity: Current position
        current_avg_entry: Current average entry price
        current_realized_pnl: Current realized P&L
        fill_quantity: Fill quantity
        fill_price: Fill price
        fill_side: Fill side (1=buy, -1=sell)
        fee: Transaction fee
        contract_multiplier: Contract multiplier

    Returns:
        (new_quantity, new_avg_entry, new_realized_pnl)
    """
    new_qty, new_avg, realized_pnl = update_position(
        current_quantity,
        current_avg_entry,
        fill_quantity,
        fill_price,
        fill_side,
        contract_multiplier
    )

    # Add realized P&L from this fill (subtract fee)
    new_realized_pnl = current_realized_pnl + realized_pnl - fee

    return new_qty, new_avg, new_realized_pnl