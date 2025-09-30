"""
Price crossing detection for fill simulation.

Detects when market prices cross our order prices between snapshots,
including complex scenarios like:
- Bid crossing up through multiple ask levels
- Ask crossing down through multiple bid levels
- Gap moves where spread widens or narrows dramatically
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class PriceCrossEvent:
    """
    Represents a price crossing event between two snapshots.

    Attributes:
        crossed: Whether a crossing occurred
        cross_direction: 1 for bid moved up, -1 for ask moved down, 0 for no cross
        volume_consumed: Estimated volume consumed during the cross
        aggressive_side: Which side was aggressive (1=buy, -1=sell)
    """
    crossed: bool
    cross_direction: int  # 1=up, -1=down, 0=none
    volume_consumed: float
    aggressive_side: int  # 1=buy, -1=sell


def detect_price_cross(
    snapshot_prev: np.ndarray,
    snapshot_curr: np.ndarray,
    order_price: float,
    order_side: int
) -> PriceCrossEvent:
    """
    Detect if market price crossed through our order price.

    Handles complex scenarios:
    1. Normal crossing: bp1_curr > ap1_prev (buy aggressive)
    2. Normal crossing: ap1_curr < bp1_prev (sell aggressive)
    3. Gap moves: Large price jumps
    4. Spread changes: Spread widening/narrowing

    Args:
        snapshot_prev: Previous snapshot
        snapshot_curr: Current snapshot
        order_price: Our order limit price
        order_side: 1 for buy, -1 for sell

    Returns:
        PriceCrossEvent with crossing details
    """
    # Extract best bid/ask from both snapshots
    bp1_prev = float(snapshot_prev['bid_price_0'])
    ap1_prev = float(snapshot_prev['ask_price_0'])
    bp1_curr = float(snapshot_curr['bid_price_0'])
    ap1_curr = float(snapshot_curr['ask_price_0'])

    # Get cumulative volume change as signal for trade intensity
    vol_prev = float(snapshot_prev['cumulative_volume'])
    vol_curr = float(snapshot_curr['cumulative_volume'])
    vol_delta = vol_curr - vol_prev

    # Check for our order side
    if order_side == 1:  # Buy order (limit buy)
        return _detect_buy_order_cross(
            bp1_prev, ap1_prev, bp1_curr, ap1_curr,
            order_price, vol_delta
        )
    else:  # Sell order (limit sell)
        return _detect_sell_order_cross(
            bp1_prev, ap1_prev, bp1_curr, ap1_curr,
            order_price, vol_delta
        )


def _detect_buy_order_cross(
    bp1_prev: float,
    ap1_prev: float,
    bp1_curr: float,
    ap1_curr: float,
    order_price: float,
    vol_delta: float
) -> PriceCrossEvent:
    """
    Detect if a buy order gets crossed.

    Buy order at `order_price` gets filled if:
    1. Market ask price moves down to or through our price
    2. Market becomes aggressive (bp crosses above prev ap)

    Scenarios:
    - Normal: ap1_curr <= order_price (market came to us)
    - Aggressive: bp1_curr > ap1_prev AND order_price was in the crossed range
    """
    # Case 1: Current ask moved down to or through our buy price
    if ap1_curr <= order_price:
        # Market came down to our price level
        # We can be filled as liquidity provider
        return PriceCrossEvent(
            crossed=True,
            cross_direction=-1,  # Ask moved down
            volume_consumed=vol_delta,
            aggressive_side=-1  # Sell side aggressive
        )

    # Case 2: Aggressive buy side crossed spread
    # Previous: bp1_prev < ap1_prev (normal spread)
    # Current: bp1_curr >= ap1_prev (bid crossed previous ask)
    if bp1_curr >= ap1_prev:
        # Aggressive buyers crossed the spread
        # Check if our buy order was in the crossed range
        # Our buy order between bp1_prev and ap1_prev would be filled
        if order_price >= bp1_prev and order_price <= bp1_curr:
            # Our order was in the path of aggressive buyers
            # This is less likely but possible (we were already best bid or inside spread)
            return PriceCrossEvent(
                crossed=True,
                cross_direction=1,  # Bid moved up aggressively
                volume_consumed=vol_delta,
                aggressive_side=1  # Buy side aggressive
            )

    # Case 3: Gap down - ask dropped significantly
    # Our buy order might have been above old ask but below new levels
    if ap1_curr < ap1_prev and order_price >= ap1_curr and order_price < ap1_prev:
        # Price gapped down through our order
        return PriceCrossEvent(
            crossed=True,
            cross_direction=-1,
            volume_consumed=vol_delta * 0.5,  # Conservative estimate
            aggressive_side=-1
        )

    # No crossing detected
    return PriceCrossEvent(
        crossed=False,
        cross_direction=0,
        volume_consumed=0.0,
        aggressive_side=0
    )


def _detect_sell_order_cross(
    bp1_prev: float,
    ap1_prev: float,
    bp1_curr: float,
    ap1_curr: float,
    order_price: float,
    vol_delta: float
) -> PriceCrossEvent:
    """
    Detect if a sell order gets crossed.

    Sell order at `order_price` gets filled if:
    1. Market bid price moves up to or through our price
    2. Market becomes aggressive (ap crosses below prev bp)

    Scenarios:
    - Normal: bp1_curr >= order_price (market came to us)
    - Aggressive: ap1_curr < bp1_prev AND order_price was in the crossed range
    """
    # Case 1: Current bid moved up to or through our sell price
    if bp1_curr >= order_price:
        # Market came up to our price level
        # We can be filled as liquidity provider
        return PriceCrossEvent(
            crossed=True,
            cross_direction=1,  # Bid moved up
            volume_consumed=vol_delta,
            aggressive_side=1  # Buy side aggressive
        )

    # Case 2: Aggressive sell side crossed spread
    # Previous: bp1_prev < ap1_prev (normal spread)
    # Current: ap1_curr <= bp1_prev (ask crossed previous bid)
    if ap1_curr <= bp1_prev:
        # Aggressive sellers crossed the spread
        # Check if our sell order was in the crossed range
        # Our sell order between ap1_prev and bp1_prev would be filled
        if order_price <= ap1_prev and order_price >= ap1_curr:
            # Our order was in the path of aggressive sellers
            # This is less likely but possible (we were already best ask or inside spread)
            return PriceCrossEvent(
                crossed=True,
                cross_direction=-1,  # Ask moved down aggressively
                volume_consumed=vol_delta,
                aggressive_side=-1  # Sell side aggressive
            )

    # Case 3: Gap up - bid jumped significantly
    # Our sell order might have been below old bid but above new levels
    if bp1_curr > bp1_prev and order_price <= bp1_curr and order_price > bp1_prev:
        # Price gapped up through our order
        return PriceCrossEvent(
            crossed=True,
            cross_direction=1,
            volume_consumed=vol_delta * 0.5,  # Conservative estimate
            aggressive_side=1
        )

    # No crossing detected
    return PriceCrossEvent(
        crossed=False,
        cross_direction=0,
        volume_consumed=0.0,
        aggressive_side=0
    )


def estimate_fill_from_cross(
    cross_event: PriceCrossEvent,
    order_price: float,
    order_quantity: float,
    order_side: int,
    snapshot_prev: np.ndarray,
    snapshot_curr: np.ndarray
) -> Optional[Tuple[float, float]]:
    """
    Estimate fill quantity and price when a crossing occurs.

    Uses:
    - Volume delta (cumulative_volume change)
    - Turnover delta (turnover_volume change) to estimate average fill price
    - Order position relative to market

    Args:
        cross_event: Detected crossing event
        order_price: Our order limit price
        order_quantity: Order quantity
        order_side: 1 for buy, -1 for sell
        snapshot_prev: Previous snapshot
        snapshot_curr: Current snapshot

    Returns:
        (fill_price, fill_quantity) if filled, else None
    """
    if not cross_event.crossed:
        return None

    # Use cumulative volume change as measure of market activity
    vol_delta = snapshot_curr['cumulative_volume'] - snapshot_prev['cumulative_volume']

    if vol_delta <= 0:
        # No volume traded - suspicious crossing, likely just spread change
        # Be conservative and don't fill
        return None

    # Calculate average trade price from turnover
    turnover_delta = snapshot_curr['turnover_volume'] - snapshot_prev['turnover_volume']
    avg_trade_price = turnover_delta / vol_delta if vol_delta > 0 else order_price

    # Determine fill price
    if cross_event.aggressive_side == order_side:
        # We were passive (providing liquidity at our limit)
        fill_price = order_price
    else:
        # We were in the path of aggressive flow
        # Use weighted average of our price and market price
        fill_price = (order_price + avg_trade_price) / 2

    # Determine fill quantity
    # Conservative: assume we get a fraction of the volume delta
    # based on how central our price was to the crossing range
    bp1_prev = float(snapshot_prev['bid_price_0'])
    ap1_prev = float(snapshot_prev['ask_price_0'])
    bp1_curr = float(snapshot_curr['bid_price_0'])
    ap1_curr = float(snapshot_curr['ask_price_0'])

    # Calculate how much of the price range we captured
    if order_side == 1:  # Buy order
        # Range: [bp1_prev, bp1_curr] or [ap1_curr, ap1_prev]
        range_low = min(bp1_prev, ap1_curr)
        range_high = max(bp1_curr, ap1_prev)
    else:  # Sell order
        # Range: [ap1_prev, ap1_curr] or [bp1_prev, bp1_curr]
        range_low = min(ap1_prev, bp1_curr)
        range_high = max(ap1_curr, bp1_prev)

    # Our capture ratio
    if range_high > range_low:
        # How favorable is our price within the range
        if order_side == 1:  # Buy
            price_ratio = (order_price - range_low) / (range_high - range_low)
        else:  # Sell
            price_ratio = (range_high - order_price) / (range_high - range_low)
        price_ratio = max(0.0, min(1.0, price_ratio))
    else:
        price_ratio = 0.5  # Default to middle

    # Estimate our fill: we get a portion of volume_delta based on our price
    # Conservative: max 50% of volume delta for single order
    our_volume_share = vol_delta * price_ratio * 0.5
    fill_quantity = min(order_quantity, our_volume_share)

    if fill_quantity > 0.01:  # Minimum fill threshold
        return fill_price, fill_quantity

    return None