"""
Order book state tracking system.

Maintains precise price-volume relationships across snapshots for accurate
fill simulation. Tracks volume changes at each price level to determine
when and how much of our orders get filled.

Uses microstructure model to decompose volume changes into trades, adds, and cancels.
"""

import numpy as np
import numba
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from .microstructure import MicrostructureModel, MicrostructureParams, VolumeChange
from .price_cross_detector import detect_price_cross, estimate_fill_from_cross


@dataclass
class PriceLevel:
    """
    State of a single price level in the order book.

    Attributes:
        price: Price of this level
        volume: Total volume at this price
        our_volume: Our order volume at this price
        our_position: Our position in queue (volume ahead of us)
    """
    price: float
    volume: float
    our_volume: float = 0.0
    our_position: float = 0.0  # Volume ahead of us in queue


class OrderBookState:
    """
    Maintains the state of the order book across snapshots.

    Tracks:
    - Volume at each price level (bid/ask)
    - Our orders and their queue positions
    - Volume changes between snapshots for fill simulation

    Uses microstructure model to decompose volume changes into:
    - Trades (consumed by market orders)
    - New orders (added to book)
    - Cancellations (before/after our position)

    This enables precise fill simulation based on actual market depth changes.
    """

    def __init__(
        self,
        microstructure_params: Optional[MicrostructureParams] = None,
        random_seed: int = 42
    ):
        """
        Initialize empty order book state.

        Args:
            microstructure_params: Parameters for microstructure model
            random_seed: Random seed for reproducibility
        """
        # Map: price -> PriceLevel
        self.bid_levels: Dict[float, PriceLevel] = {}
        self.ask_levels: Dict[float, PriceLevel] = {}

        # Previous snapshot for comparison
        self.prev_snapshot: Optional[np.ndarray] = None

        # Microstructure model for decomposing volume changes
        self.microstructure = MicrostructureModel(microstructure_params)
        self.random_state = np.random.RandomState(random_seed)

    def update_from_snapshot(self, snapshot: np.ndarray) -> None:
        """
        Update order book state from a new snapshot.

        Args:
            snapshot: New snapshot data
        """
        # Extract bid and ask data
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
        for price, volume in zip(bid_prices, bid_volumes):
            if price > 0 and volume > 0:  # Valid level
                if price in self.bid_levels:
                    # Existing level - preserve our order info
                    old_level = self.bid_levels[price]
                    new_bid_levels[price] = PriceLevel(
                        price=price,
                        volume=volume,
                        our_volume=old_level.our_volume,
                        our_position=old_level.our_position
                    )
                else:
                    # New level
                    new_bid_levels[price] = PriceLevel(price=price, volume=volume)

        # Update ask levels
        new_ask_levels = {}
        for price, volume in zip(ask_prices, ask_volumes):
            if price > 0 and volume > 0:  # Valid level
                if price in self.ask_levels:
                    # Existing level - preserve our order info
                    old_level = self.ask_levels[price]
                    new_ask_levels[price] = PriceLevel(
                        price=price,
                        volume=volume,
                        our_volume=old_level.our_volume,
                        our_position=old_level.our_position
                    )
                else:
                    # New level
                    new_ask_levels[price] = PriceLevel(price=price, volume=volume)

        self.bid_levels = new_bid_levels
        self.ask_levels = new_ask_levels
        self.prev_snapshot = snapshot

    def add_our_order(
        self,
        price: float,
        quantity: float,
        side: int
    ) -> Tuple[float, float]:
        """
        Add our order to the order book and calculate initial queue position.

        Args:
            price: Order price
            quantity: Order quantity
            side: 1 for buy (bid), -1 for sell (ask)

        Returns:
            (queue_position, volume_ahead): Our position in the queue
        """
        levels = self.bid_levels if side == 1 else self.ask_levels

        if price not in levels:
            # Price not in current book - we're alone at this level
            # This happens when posting outside current 5 levels
            level = PriceLevel(
                price=price,
                volume=quantity,
                our_volume=quantity,
                our_position=0.0
            )
            levels[price] = level
            return 0.0, 0.0

        # Join back of queue at existing price
        level = levels[price]
        volume_ahead = level.volume  # All current volume is ahead of us
        level.our_volume += quantity
        level.our_position = volume_ahead
        level.volume += quantity  # Add our volume to total

        return volume_ahead, volume_ahead

    def calculate_volume_traded_at_price(
        self,
        snapshot_prev: np.ndarray,
        snapshot_curr: np.ndarray,
        price: float,
        side: int
    ) -> float:
        """
        Calculate precise volume traded at a specific price level.

        Uses order book state to determine volume consumed between snapshots.

        Args:
            snapshot_prev: Previous snapshot
            snapshot_curr: Current snapshot
            price: Price level
            side: 1 for bid, -1 for ask

        Returns:
            Volume traded at this price (consumed from front of queue)
        """
        # Get volume at this price in both snapshots
        vol_prev = self._get_snapshot_volume(snapshot_prev, price, side)
        vol_curr = self._get_snapshot_volume(snapshot_curr, price, side)

        # If price not in prev snapshot, no volume traded at this price
        if vol_prev == 0.0:
            return 0.0

        # If price disappeared from book, consider all volume as traded
        if vol_prev > 0 and vol_curr == 0.0:
            return vol_prev

        # Volume decrease indicates trades
        if vol_prev > vol_curr:
            volume_traded = vol_prev - vol_curr
            return volume_traded

        return 0.0

    def _get_snapshot_volume(
        self,
        snapshot: np.ndarray,
        price: float,
        side: int
    ) -> float:
        """Get volume at price from snapshot."""
        if side == 1:  # Bid
            prices = [snapshot['bid_price_0'], snapshot['bid_price_1'],
                     snapshot['bid_price_2'], snapshot['bid_price_3'],
                     snapshot['bid_price_4']]
            volumes = [snapshot['bid_volume_0'], snapshot['bid_volume_1'],
                      snapshot['bid_volume_2'], snapshot['bid_volume_3'],
                      snapshot['bid_volume_4']]
        else:  # Ask
            prices = [snapshot['ask_price_0'], snapshot['ask_price_1'],
                     snapshot['ask_price_2'], snapshot['ask_price_3'],
                     snapshot['ask_price_4']]
            volumes = [snapshot['ask_volume_0'], snapshot['ask_volume_1'],
                      snapshot['ask_volume_2'], snapshot['ask_volume_3'],
                      snapshot['ask_volume_4']]

        for p, v in zip(prices, volumes):
            if abs(p - price) < 1e-9:
                return v

        return 0.0

    def update_queue_position(
        self,
        order_price: float,
        order_side: int,
        current_position: float,
        snapshot_prev: np.ndarray,
        snapshot_curr: np.ndarray
    ) -> Tuple[float, float, Optional[VolumeChange]]:
        """
        Update our order's queue position using microstructure decomposition.

        Decomposes volume changes into:
        - Trades (advance our position from front)
        - Cancellations before us (advance our position)
        - Cancellations after us (no effect on position)
        - New orders (join back of queue)

        Args:
            order_price: Our order price
            order_side: 1 for bid, -1 for ask
            current_position: Current volume ahead of us
            snapshot_prev: Previous snapshot
            snapshot_curr: Current snapshot

        Returns:
            (new_position, new_volume_ahead, volume_change_detail)
        """
        levels = self.bid_levels if order_side == 1 else self.ask_levels

        if order_price not in levels:
            # Order price no longer in book
            return current_position, current_position, None

        # Use microstructure model to decompose volume change
        change = self.microstructure.decompose_volume_change(
            snapshot_prev,
            snapshot_curr,
            order_price,
            order_side,
            current_position,
            self.random_state
        )

        # Calculate new position based on decomposed components
        new_position = self.microstructure.estimate_queue_advancement(
            change,
            current_position
        )

        # Update level state
        level = levels[order_price]
        level.our_position = new_position

        return new_position, new_position, change

    def check_fill(
        self,
        order_price: float,
        order_quantity: float,
        order_side: int,
        volume_ahead: float,
        volume_traded: float,
        snapshot_prev: np.ndarray,
        snapshot_curr: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """
        Check if our order gets filled based on queue position and volume traded.

        Uses sophisticated price crossing detection to handle:
        - Orders inside the book (queue-based filling)
        - Orders outside the book (price crossing)
        - Complex scenarios: bp1_curr > ap1_prev (aggressive crossing)
        - Gap moves and spread changes

        Args:
            order_price: Our order price
            order_quantity: Remaining order quantity
            order_side: 1 for bid, -1 for ask
            volume_ahead: Volume ahead of us in queue
            volume_traded: Volume that traded at this price
            snapshot_prev: Previous snapshot (for crossing detection)
            snapshot_curr: Current snapshot

        Returns:
            (fill_price, fill_quantity) if filled, else None
        """
        # Check if order is in current book
        order_in_book = self._get_snapshot_volume(snapshot_curr, order_price, order_side) > 0

        if not order_in_book:
            # Order not in visible book - use advanced price crossing detection
            cross_event = detect_price_cross(
                snapshot_prev,
                snapshot_curr,
                order_price,
                order_side
            )

            if cross_event.crossed:
                # Price crossed our order - estimate fill
                return estimate_fill_from_cross(
                    cross_event,
                    order_price,
                    order_quantity,
                    order_side,
                    snapshot_prev,
                    snapshot_curr
                )
            return None

        # Order is in book - use queue-based fill simulation
        if volume_traded <= 0:
            return None

        # Check if volume traded clears queue ahead of us
        if volume_traded <= volume_ahead:
            # Not reached us yet
            return None

        # Queue cleared - we can be filled
        available_volume = volume_traded - volume_ahead
        fill_quantity = min(order_quantity, available_volume)

        if fill_quantity > 0:
            return order_price, fill_quantity

        return None

    def remove_order(
        self,
        price: float,
        quantity: float,
        side: int
    ) -> None:
        """
        Remove our order from the book (after fill or cancel).

        Args:
            price: Order price
            quantity: Quantity to remove
            side: 1 for bid, -1 for ask
        """
        levels = self.bid_levels if side == 1 else self.ask_levels

        if price in levels:
            level = levels[price]
            level.our_volume = max(0.0, level.our_volume - quantity)
            # Note: We don't adjust total volume as that comes from market snapshots


@numba.jit(nopython=True, cache=True)
def calculate_queue_advancement(
    initial_volume_ahead: float,
    volume_traded: float,
    cancellation_rate: float,
    random_value: float
) -> float:
    """
    Calculate new queue position with Numba optimization.

    Args:
        initial_volume_ahead: Current position in queue
        volume_traded: Volume that traded
        cancellation_rate: Probability of cancellation
        random_value: Random value [0, 1)

    Returns:
        New volume ahead of us
    """
    # Deterministic advancement
    volume_ahead = max(0.0, initial_volume_ahead - volume_traded)

    # Probabilistic cancellation
    if volume_ahead > 0 and cancellation_rate > 0:
        expected_cancelled = volume_ahead * cancellation_rate
        std_dev = np.sqrt(volume_ahead * cancellation_rate * (1 - cancellation_rate))
        z_score = (random_value - 0.5) * 4
        cancelled = expected_cancelled + z_score * std_dev
        cancelled = max(0.0, min(volume_ahead, cancelled))
        volume_ahead = max(0.0, volume_ahead - cancelled)

    return volume_ahead