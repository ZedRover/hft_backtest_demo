"""
Order book management with snapshot alignment.

Maintains a clean market orderbook (without our orders) by:
1. Taking snapshots from the market
2. Subtracting our active orders from the snapshot volumes
3. Using the clean orderbook for fill simulation

This is critical because:
- Snapshot volumes include ALL participants (including us)
- We need to know the "market volume" to calculate queue positions
- Our orders change the snapshot volumes
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field


@dataclass
class OrderBookLevel:
    """
    A single price level in the order book.

    Attributes:
        price: Price of this level
        market_volume: Volume from other participants (clean)
        our_volume: Our total volume at this price
        our_orders: List of (order_id, quantity, position) tuples
    """
    price: float
    market_volume: float  # Volume from market (without us)
    our_volume: float = 0.0  # Our total volume
    our_orders: List[Tuple[int, float, float]] = field(default_factory=list)


class OrderBook:
    """
    Maintains a clean market orderbook aligned with snapshots.

    Key concept:
    - snapshot_volume = market_volume + our_volume
    - We extract market_volume by subtracting our orders
    - Use market_volume for queue position calculations
    """

    def __init__(self):
        """Initialize empty orderbook."""
        self.bid_levels: Dict[float, OrderBookLevel] = {}
        self.ask_levels: Dict[float, OrderBookLevel] = {}

        # Track all our active orders: order_id -> (price, side, quantity)
        self.our_active_orders: Dict[int, Tuple[float, int, float]] = {}

    def update_from_snapshot(self, snapshot: np.ndarray) -> None:
        """
        Update orderbook from snapshot, extracting clean market volumes.

        Process:
        1. Read snapshot volumes (includes everyone)
        2. Subtract our active orders from those volumes
        3. Store clean market volumes

        Args:
            snapshot: Market snapshot (includes our orders)
        """
        # Extract bid data
        bid_prices = np.array([
            snapshot['bid_price_0'], snapshot['bid_price_1'],
            snapshot['bid_price_2'], snapshot['bid_price_3'],
            snapshot['bid_price_4']
        ])
        bid_volumes = np.array([
            snapshot['bid_volume_0'], snapshot['bid_volume_1'],
            snapshot['bid_volume_2'], snapshot['bid_volume_3'],
            snapshot['bid_volume_4']
        ])

        # Extract ask data
        ask_prices = np.array([
            snapshot['ask_price_0'], snapshot['ask_price_1'],
            snapshot['ask_price_2'], snapshot['ask_price_3'],
            snapshot['ask_price_4']
        ])
        ask_volumes = np.array([
            snapshot['ask_volume_0'], snapshot['ask_volume_1'],
            snapshot['ask_volume_2'], snapshot['ask_volume_3'],
            snapshot['ask_volume_4']
        ])

        # Update bid levels
        new_bid_levels = {}
        for price, snapshot_vol in zip(bid_prices, bid_volumes):
            if price > 0 and snapshot_vol > 0:
                # Calculate market volume by subtracting our orders
                our_vol = self._get_our_volume_at_price(price, side=1)
                market_vol = max(0.0, snapshot_vol - our_vol)

                # Preserve existing order positions if price exists
                if price in self.bid_levels:
                    old_level = self.bid_levels[price]
                    new_bid_levels[price] = OrderBookLevel(
                        price=price,
                        market_volume=market_vol,
                        our_volume=our_vol,
                        our_orders=old_level.our_orders.copy()
                    )
                else:
                    new_bid_levels[price] = OrderBookLevel(
                        price=price,
                        market_volume=market_vol,
                        our_volume=our_vol
                    )

        # Update ask levels
        new_ask_levels = {}
        for price, snapshot_vol in zip(ask_prices, ask_volumes):
            if price > 0 and snapshot_vol > 0:
                # Calculate market volume by subtracting our orders
                our_vol = self._get_our_volume_at_price(price, side=-1)
                market_vol = max(0.0, snapshot_vol - our_vol)

                # Preserve existing order positions if price exists
                if price in self.ask_levels:
                    old_level = self.ask_levels[price]
                    new_ask_levels[price] = OrderBookLevel(
                        price=price,
                        market_volume=market_vol,
                        our_volume=our_vol,
                        our_orders=old_level.our_orders.copy()
                    )
                else:
                    new_ask_levels[price] = OrderBookLevel(
                        price=price,
                        market_volume=market_vol,
                        our_volume=our_vol
                    )

        self.bid_levels = new_bid_levels
        self.ask_levels = new_ask_levels

    def add_our_order(
        self,
        order_id: int,
        price: float,
        quantity: float,
        side: int
    ) -> Tuple[float, float]:
        """
        Add our order to the orderbook.

        Args:
            order_id: Unique order ID
            price: Order price
            quantity: Order quantity
            side: 1 for bid, -1 for ask

        Returns:
            (queue_position, volume_ahead): Position in queue
        """
        levels = self.bid_levels if side == 1 else self.ask_levels

        # Track this order
        self.our_active_orders[order_id] = (price, side, quantity)

        if price not in levels:
            # Price not in visible book - we're creating a new level
            level = OrderBookLevel(
                price=price,
                market_volume=0.0,  # No market volume at this price
                our_volume=quantity
            )
            level.our_orders.append((order_id, quantity, 0.0))
            levels[price] = level
            return 0.0, 0.0

        # Join back of queue at existing price
        level = levels[price]
        volume_ahead = level.market_volume  # All market volume is ahead

        # Add our order info
        level.our_orders.append((order_id, quantity, volume_ahead))
        level.our_volume += quantity

        return volume_ahead, volume_ahead

    def remove_our_order(
        self,
        order_id: int,
        quantity: float
    ) -> None:
        """
        Remove our order (after fill or cancel).

        Args:
            order_id: Order ID to remove
            quantity: Quantity to remove (for partial fills)
        """
        if order_id not in self.our_active_orders:
            return

        price, side, order_qty = self.our_active_orders[order_id]
        levels = self.bid_levels if side == 1 else self.ask_levels

        if price in levels:
            level = levels[price]
            level.our_volume = max(0.0, level.our_volume - quantity)

            # Remove from our_orders list
            level.our_orders = [
                (oid, qty, pos) for oid, qty, pos in level.our_orders
                if oid != order_id
            ]

        # If fully removed, delete from tracking
        if quantity >= order_qty:
            del self.our_active_orders[order_id]
        else:
            # Partial fill - update quantity
            self.our_active_orders[order_id] = (price, side, order_qty - quantity)

    def get_market_volume_at_price(
        self,
        price: float,
        side: int
    ) -> float:
        """
        Get clean market volume at a price (without our orders).

        Args:
            price: Price level
            side: 1 for bid, -1 for ask

        Returns:
            Market volume (excluding us)
        """
        levels = self.bid_levels if side == 1 else self.ask_levels
        if price in levels:
            return levels[price].market_volume
        return 0.0

    def get_our_queue_position(
        self,
        order_id: int
    ) -> Optional[Tuple[float, float]]:
        """
        Get our order's queue position.

        Args:
            order_id: Order ID

        Returns:
            (volume_ahead, our_quantity) or None if not found
        """
        if order_id not in self.our_active_orders:
            return None

        price, side, quantity = self.our_active_orders[order_id]
        levels = self.bid_levels if side == 1 else self.ask_levels

        if price not in levels:
            return None

        level = levels[price]

        # Find this order in the level
        for oid, qty, position in level.our_orders:
            if oid == order_id:
                return position, qty

        return None

    def update_order_queue_position(
        self,
        order_id: int,
        new_position: float
    ) -> None:
        """
        Update an order's queue position.

        Args:
            order_id: Order ID
            new_position: New volume ahead
        """
        if order_id not in self.our_active_orders:
            return

        price, side, quantity = self.our_active_orders[order_id]
        levels = self.bid_levels if side == 1 else self.ask_levels

        if price in levels:
            level = levels[price]
            # Update position in our_orders list
            level.our_orders = [
                (oid, qty, new_position if oid == order_id else pos)
                for oid, qty, pos in level.our_orders
            ]

    def _get_our_volume_at_price(self, price: float, side: int) -> float:
        """Get total volume of our orders at a price."""
        total = 0.0
        for order_id, (order_price, order_side, qty) in self.our_active_orders.items():
            if abs(order_price - price) < 1e-9 and order_side == side:
                total += qty
        return total

    def get_snapshot_volume(
        self,
        snapshot: np.ndarray,
        price: float,
        side: int
    ) -> float:
        """
        Get total volume at price from snapshot (includes everyone).

        Args:
            snapshot: Market snapshot
            price: Price level
            side: 1 for bid, -1 for ask

        Returns:
            Total volume from snapshot
        """
        if side == 1:  # Bid
            prices = [
                snapshot['bid_price_0'], snapshot['bid_price_1'],
                snapshot['bid_price_2'], snapshot['bid_price_3'],
                snapshot['bid_price_4']
            ]
            volumes = [
                snapshot['bid_volume_0'], snapshot['bid_volume_1'],
                snapshot['bid_volume_2'], snapshot['bid_volume_3'],
                snapshot['bid_volume_4']
            ]
        else:  # Ask
            prices = [
                snapshot['ask_price_0'], snapshot['ask_price_1'],
                snapshot['ask_price_2'], snapshot['ask_price_3'],
                snapshot['ask_price_4']
            ]
            volumes = [
                snapshot['ask_volume_0'], snapshot['ask_volume_1'],
                snapshot['ask_volume_2'], snapshot['ask_volume_3'],
                snapshot['ask_volume_4']
            ]

        for p, v in zip(prices, volumes):
            if abs(p - price) < 1e-9:
                return float(v)

        return 0.0


# Helper functions for backward compatibility

def get_mid_price(snapshot: np.ndarray) -> float:
    """
    Calculate mid price from snapshot.

    Args:
        snapshot: Market snapshot

    Returns:
        Mid price (bid + ask) / 2
    """
    bid = float(snapshot['bid_price_0'])
    ask = float(snapshot['ask_price_0'])
    return (bid + ask) / 2.0


def get_spread(snapshot: np.ndarray) -> float:
    """
    Calculate bid-ask spread from snapshot.

    Args:
        snapshot: Single snapshot record

    Returns:
        Spread (best_ask - best_bid)
    """
    best_bid = snapshot['bid_price_0']
    best_ask = snapshot['ask_price_0']
    return best_ask - best_bid


def get_bid_prices(snapshot: np.ndarray) -> np.ndarray:
    """Extract all bid prices from snapshot."""
    return np.array([
        snapshot['bid_price_0'],
        snapshot['bid_price_1'],
        snapshot['bid_price_2'],
        snapshot['bid_price_3'],
        snapshot['bid_price_4'],
    ], dtype=np.float64)


def get_ask_prices(snapshot: np.ndarray) -> np.ndarray:
    """Extract all ask prices from snapshot."""
    return np.array([
        snapshot['ask_price_0'],
        snapshot['ask_price_1'],
        snapshot['ask_price_2'],
        snapshot['ask_price_3'],
        snapshot['ask_price_4'],
    ], dtype=np.float64)


def get_bid_volumes(snapshot: np.ndarray) -> np.ndarray:
    """Extract all bid volumes from snapshot."""
    return np.array([
        snapshot['bid_volume_0'],
        snapshot['bid_volume_1'],
        snapshot['bid_volume_2'],
        snapshot['bid_volume_3'],
        snapshot['bid_volume_4'],
    ], dtype=np.float64)


def get_ask_volumes(snapshot: np.ndarray) -> np.ndarray:
    """Extract all ask volumes from snapshot."""
    return np.array([
        snapshot['ask_volume_0'],
        snapshot['ask_volume_1'],
        snapshot['ask_volume_2'],
        snapshot['ask_volume_3'],
        snapshot['ask_volume_4'],
    ], dtype=np.float64)


def find_price_level(
    snapshot: np.ndarray,
    price: float,
    side: int
) -> Tuple[int, bool]:
    """
    Find which level a price corresponds to in the order book.

    Args:
        snapshot: Single snapshot record
        price: Price to find
        side: 1 for bid, -1 for ask

    Returns:
        (level, found) where level is 0-4 and found indicates if price exists
    """
    if side == 1:  # Bid
        prices = get_bid_prices(snapshot)
    else:  # Ask
        prices = get_ask_prices(snapshot)

    # Find exact match
    matches = np.where(np.abs(prices - price) < 1e-9)[0]
    if len(matches) > 0:
        return int(matches[0]), True

    return -1, False


def get_volume_at_price(
    snapshot: np.ndarray,
    price: float,
    side: int
) -> float:
    """
    Get volume available at a specific price level.

    Args:
        snapshot: Single snapshot record
        price: Price level
        side: 1 for bid, -1 for ask

    Returns:
        Volume at that price (0.0 if price not found)
    """
    level, found = find_price_level(snapshot, price, side)

    if not found:
        return 0.0

    if side == 1:  # Bid
        volumes = get_bid_volumes(snapshot)
    else:  # Ask
        volumes = get_ask_volumes(snapshot)

    return volumes[level]


def calculate_volume_traded_at_price(
    snapshot_prev: np.ndarray,
    snapshot_curr: np.ndarray,
    price: float,
    side: int
) -> float:
    """
    Calculate volume traded at a specific price between snapshots.

    Note: This uses snapshot volumes which include all participants.
    For queue simulation, use OrderBook.get_market_volume_at_price()

    Args:
        snapshot_prev: Previous snapshot
        snapshot_curr: Current snapshot
        price: Price level
        side: 1 for bid, -1 for ask

    Returns:
        Volume traded (consumed)
    """
    vol_prev = get_volume_at_price(snapshot_prev, price, side)
    vol_curr = get_volume_at_price(snapshot_curr, price, side)

    if vol_prev == 0.0:
        return 0.0

    if vol_prev > 0 and vol_curr == 0.0:
        return vol_prev

    if vol_prev > vol_curr:
        return vol_prev - vol_curr

    return 0.0