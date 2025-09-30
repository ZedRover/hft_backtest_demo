"""
Unified order book implementation.

Merges the functionality of OrderBook and OrderBookState into a single,
cohesive component that:
1. Tracks clean market state (separating our orders from market volumes)
2. Uses microstructure model for queue position updates
3. Handles fill simulation with both queue-based and price-crossing logic

This eliminates redundancy and provides a single source of truth for order book state.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

from .microstructure import MicrostructureModel, MicrostructureParams, VolumeChange
from .constants import (
    OrderSide,
    PRICE_EPSILON,
    ORDER_BOOK_DEPTH,
    MIN_FILL_QUANTITY
)
from .exceptions import (
    OrderNotFoundError,
    PriceNotFoundError,
    InvalidSnapshotError
)


@dataclass
class OrderBookLevel:
    """
    A single price level in the unified order book.
    
    Attributes:
        price: Price of this level
        market_volume: Volume from other market participants (clean)
        our_total_volume: Total volume from our orders at this price
        our_orders: List of (order_id, quantity, volume_ahead) tuples
    """
    price: float
    market_volume: float
    our_total_volume: float = 0.0
    our_orders: List[Tuple[int, float, float]] = field(default_factory=list)
    
    @property
    def total_volume(self) -> float:
        """Total volume at this price (market + ours)."""
        return self.market_volume + self.our_total_volume


class UnifiedOrderBook:
    """
    Unified order book that combines market state tracking and fill simulation.
    
    Key features:
    - Maintains clean market volumes (without our orders)
    - Tracks our order positions in the queue
    - Uses microstructure model for realistic queue advancement
    - Handles both queue-based and price-crossing fills
    - Single source of truth for order book state
    
    Design:
    - Implements OrderBookProtocol for dependency injection
    - Integrates microstructure model for volume decomposition
    - Provides both order book and fill simulation capabilities
    """
    
    def __init__(
        self,
        microstructure_params: Optional[MicrostructureParams] = None,
        random_seed: int = 42
    ):
        """
        Initialize unified order book.
        
        Args:
            microstructure_params: Parameters for microstructure model
            random_seed: Random seed for reproducibility
        """
        # Order book state
        self.bid_levels: Dict[float, OrderBookLevel] = {}
        self.ask_levels: Dict[float, OrderBookLevel] = {}
        
        # Track our active orders: order_id -> (price, side, quantity)
        self.our_active_orders: Dict[int, Tuple[float, int, float]] = {}
        
        # Microstructure model for queue simulation
        self.microstructure = MicrostructureModel(microstructure_params)
        self.random_state = np.random.RandomState(random_seed)
        
        # Previous snapshot for comparison
        self.prev_snapshot: Optional[np.ndarray] = None
    
    def update_from_snapshot(self, snapshot: np.ndarray) -> None:
        """
        Update order book from market snapshot.
        
        Extracts clean market volumes by subtracting our orders from snapshot volumes.
        
        Args:
            snapshot: Market snapshot (includes all participants)
            
        Raises:
            InvalidSnapshotError: If snapshot is malformed
        """
        try:
            # Extract bid data
            bid_prices = np.array([
                snapshot['bid_price_0'], snapshot['bid_price_1'],
                snapshot['bid_price_2'], snapshot['bid_price_3'],
                snapshot['bid_price_4']
            ], dtype=np.float64)
            bid_volumes = np.array([
                snapshot['bid_volume_0'], snapshot['bid_volume_1'],
                snapshot['bid_volume_2'], snapshot['bid_volume_3'],
                snapshot['bid_volume_4']
            ], dtype=np.float64)
            
            # Extract ask data
            ask_prices = np.array([
                snapshot['ask_price_0'], snapshot['ask_price_1'],
                snapshot['ask_price_2'], snapshot['ask_price_3'],
                snapshot['ask_price_4']
            ], dtype=np.float64)
            ask_volumes = np.array([
                snapshot['ask_volume_0'], snapshot['ask_volume_1'],
                snapshot['ask_volume_2'], snapshot['ask_volume_3'],
                snapshot['ask_volume_4']
            ], dtype=np.float64)
        except (KeyError, IndexError) as e:
            timestamp = int(snapshot['timestamp']) if 'timestamp' in snapshot.dtype.names else 0
            raise InvalidSnapshotError(timestamp, f"Missing required field: {e}")
        
        # Update bid levels
        new_bid_levels = {}
        for price, snapshot_vol in zip(bid_prices, bid_volumes):
            if price > 0 and snapshot_vol > 0:
                # Calculate clean market volume
                our_vol = self._get_our_volume_at_price(price, OrderSide.BID)
                market_vol = max(0.0, snapshot_vol - our_vol)
                
                # Preserve existing order positions if level exists
                if price in self.bid_levels:
                    old_level = self.bid_levels[price]
                    new_bid_levels[price] = OrderBookLevel(
                        price=price,
                        market_volume=market_vol,
                        our_total_volume=our_vol,
                        our_orders=old_level.our_orders.copy()
                    )
                else:
                    new_bid_levels[price] = OrderBookLevel(
                        price=price,
                        market_volume=market_vol,
                        our_total_volume=our_vol
                    )
        
        # Update ask levels
        new_ask_levels = {}
        for price, snapshot_vol in zip(ask_prices, ask_volumes):
            if price > 0 and snapshot_vol > 0:
                # Calculate clean market volume
                our_vol = self._get_our_volume_at_price(price, OrderSide.ASK)
                market_vol = max(0.0, snapshot_vol - our_vol)
                
                # Preserve existing order positions if level exists
                if price in self.ask_levels:
                    old_level = self.ask_levels[price]
                    new_ask_levels[price] = OrderBookLevel(
                        price=price,
                        market_volume=market_vol,
                        our_total_volume=our_vol,
                        our_orders=old_level.our_orders.copy()
                    )
                else:
                    new_ask_levels[price] = OrderBookLevel(
                        price=price,
                        market_volume=market_vol,
                        our_total_volume=our_vol
                    )
        
        self.bid_levels = new_bid_levels
        self.ask_levels = new_ask_levels
        self.prev_snapshot = snapshot
    
    def add_our_order(
        self,
        order_id: int,
        price: float,
        quantity: float,
        side: int
    ) -> Tuple[float, float]:
        """
        Add our order to the order book.
        
        Args:
            order_id: Unique order identifier
            price: Order limit price
            quantity: Order quantity
            side: 1 for bid, -1 for ask
            
        Returns:
            Tuple of (volume_ahead, queue_position)
        """
        levels = self.bid_levels if side == OrderSide.BID else self.ask_levels
        
        # Track this order
        self.our_active_orders[order_id] = (price, side, quantity)
        
        if price not in levels:
            # Price not in visible book - creating new level
            level = OrderBookLevel(
                price=price,
                market_volume=0.0,
                our_total_volume=quantity
            )
            level.our_orders.append((order_id, quantity, 0.0))
            levels[price] = level
            return 0.0, 0.0
        
        # Join back of queue at existing price
        level = levels[price]
        volume_ahead = level.market_volume + level.our_total_volume
        
        # Add our order
        level.our_orders.append((order_id, quantity, volume_ahead))
        level.our_total_volume += quantity
        
        return volume_ahead, volume_ahead
    
    def remove_our_order(self, order_id: int, quantity: float) -> None:
        """
        Remove our order from the order book.
        
        Args:
            order_id: Order ID to remove
            quantity: Quantity to remove (for partial fills)
            
        Raises:
            OrderNotFoundError: If order_id not found
        """
        if order_id not in self.our_active_orders:
            raise OrderNotFoundError(order_id)
        
        price, side, order_qty = self.our_active_orders[order_id]
        levels = self.bid_levels if side == OrderSide.BID else self.ask_levels
        
        if price in levels:
            level = levels[price]
            level.our_total_volume = max(0.0, level.our_total_volume - quantity)
            
            # Remove or update in our_orders list
            new_orders = []
            for oid, qty, pos in level.our_orders:
                if oid == order_id:
                    if quantity < qty:
                        # Partial fill - keep with updated quantity
                        new_orders.append((oid, qty - quantity, pos))
                else:
                    new_orders.append((oid, qty, pos))
            level.our_orders = new_orders
        
        # Update tracking
        if quantity >= order_qty - MIN_FILL_QUANTITY:
            # Fully removed
            del self.our_active_orders[order_id]
        else:
            # Partial fill
            self.our_active_orders[order_id] = (price, side, order_qty - quantity)
    
    def get_market_volume_at_price(self, price: float, side: int) -> float:
        """
        Get clean market volume at price (excluding our orders).
        
        Args:
            price: Price level
            side: 1 for bid, -1 for ask
            
        Returns:
            Market volume at that price
        """
        levels = self.bid_levels if side == OrderSide.BID else self.ask_levels
        if price in levels:
            return levels[price].market_volume
        return 0.0
    
    def update_order_queue_position(
        self,
        order_id: int,
        new_position: float
    ) -> None:
        """
        Update an order's queue position.
        
        Args:
            order_id: Order to update
            new_position: New volume ahead in queue
            
        Raises:
            OrderNotFoundError: If order_id not found
        """
        if order_id not in self.our_active_orders:
            raise OrderNotFoundError(order_id)
        
        price, side, quantity = self.our_active_orders[order_id]
        levels = self.bid_levels if side == OrderSide.BID else self.ask_levels
        
        if price in levels:
            level = levels[price]
            # Update position in our_orders list
            level.our_orders = [
                (oid, qty, new_position if oid == order_id else pos)
                for oid, qty, pos in level.our_orders
            ]
    
    def update_queue_position_with_microstructure(
        self,
        order_id: int,
        snapshot_prev: np.ndarray,
        snapshot_curr: np.ndarray
    ) -> Tuple[float, Optional[VolumeChange]]:
        """
        Update order's queue position using microstructure decomposition.
        
        Args:
            order_id: Order to update
            snapshot_prev: Previous snapshot
            snapshot_curr: Current snapshot
            
        Returns:
            Tuple of (new_volume_ahead, volume_change_detail)
            
        Raises:
            OrderNotFoundError: If order_id not found
        """
        if order_id not in self.our_active_orders:
            raise OrderNotFoundError(order_id)
        
        price, side, quantity = self.our_active_orders[order_id]
        levels = self.bid_levels if side == OrderSide.BID else self.ask_levels
        
        if price not in levels:
            # Order price no longer in visible book
            return 0.0, None
        
        level = levels[price]
        
        # Find current position for this order
        current_position = 0.0
        for oid, qty, pos in level.our_orders:
            if oid == order_id:
                current_position = pos
                break
        
        # Use microstructure model to decompose volume change
        change = self.microstructure.decompose_volume_change(
            snapshot_prev,
            snapshot_curr,
            price,
            side,
            current_position,
            self.random_state
        )
        
        # Calculate new position based on decomposition
        new_position = self.microstructure.estimate_queue_advancement(
            change,
            current_position
        )
        
        # Update position
        self.update_order_queue_position(order_id, new_position)
        
        return new_position, change
    
    def calculate_volume_traded_at_price(
        self,
        snapshot_prev: np.ndarray,
        snapshot_curr: np.ndarray,
        price: float,
        side: int
    ) -> float:
        """
        Calculate volume traded at a specific price level.
        
        Args:
            snapshot_prev: Previous snapshot
            snapshot_curr: Current snapshot
            price: Price level
            side: 1 for bid, -1 for ask
            
        Returns:
            Volume traded (consumed) at that price
        """
        vol_prev = self._get_snapshot_volume(snapshot_prev, price, side)
        vol_curr = self._get_snapshot_volume(snapshot_curr, price, side)
        
        if vol_prev <= PRICE_EPSILON:
            return 0.0
        
        # Volume disappeared completely
        if vol_prev > PRICE_EPSILON and vol_curr <= PRICE_EPSILON:
            return vol_prev
        
        # Volume decreased
        if vol_prev > vol_curr:
            return vol_prev - vol_curr
        
        return 0.0
    
    def check_fill(
        self,
        order_id: int,
        snapshot_prev: np.ndarray,
        snapshot_curr: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """
        Check if order should be filled.
        
        Uses queue-based fill logic: if volume traded exceeds volume ahead,
        the order gets filled.
        
        Args:
            order_id: Order to check
            snapshot_prev: Previous snapshot
            snapshot_curr: Current snapshot
            
        Returns:
            Tuple of (fill_price, fill_quantity) if filled, None otherwise
            
        Raises:
            OrderNotFoundError: If order_id not found
        """
        if order_id not in self.our_active_orders:
            raise OrderNotFoundError(order_id)
        
        price, side, remaining_qty = self.our_active_orders[order_id]
        levels = self.bid_levels if side == OrderSide.BID else self.ask_levels
        
        if price not in levels:
            # Order not in visible book
            return None
        
        level = levels[price]
        
        # Find volume ahead for this order
        volume_ahead = 0.0
        for oid, qty, pos in level.our_orders:
            if oid == order_id:
                volume_ahead = pos
                break
        
        # Calculate volume traded at this price
        volume_traded = self.calculate_volume_traded_at_price(
            snapshot_prev,
            snapshot_curr,
            price,
            side
        )
        
        if volume_traded <= volume_ahead:
            # Not reached us yet
            return None
        
        # Queue cleared - we can be filled
        available_volume = volume_traded - volume_ahead
        fill_quantity = min(remaining_qty, available_volume)
        
        if fill_quantity > MIN_FILL_QUANTITY:
            return price, fill_quantity
        
        return None
    
    def _get_our_volume_at_price(self, price: float, side: int) -> float:
        """Get total volume of our orders at a specific price."""
        total = 0.0
        for order_id, (order_price, order_side, qty) in self.our_active_orders.items():
            if abs(order_price - price) < PRICE_EPSILON and order_side == side:
                total += qty
        return total
    
    def _get_snapshot_volume(
        self,
        snapshot: np.ndarray,
        price: float,
        side: int
    ) -> float:
        """Get volume at price from snapshot."""
        if side == OrderSide.BID:
            prices = [snapshot['bid_price_0'], snapshot['bid_price_1'],
                     snapshot['bid_price_2'], snapshot['bid_price_3'],
                     snapshot['bid_price_4']]
            volumes = [snapshot['bid_volume_0'], snapshot['bid_volume_1'],
                      snapshot['bid_volume_2'], snapshot['bid_volume_3'],
                      snapshot['bid_volume_4']]
        else:
            prices = [snapshot['ask_price_0'], snapshot['ask_price_1'],
                     snapshot['ask_price_2'], snapshot['ask_price_3'],
                     snapshot['ask_price_4']]
            volumes = [snapshot['ask_volume_0'], snapshot['ask_volume_1'],
                      snapshot['ask_volume_2'], snapshot['ask_volume_3'],
                      snapshot['ask_volume_4']]
        
        for p, v in zip(prices, volumes):
            if abs(p - price) < PRICE_EPSILON:
                return float(v)
        
        return 0.0


# Helper functions for backward compatibility

def get_mid_price(snapshot: np.ndarray) -> float:
    """Calculate mid price from snapshot."""
    bid = float(snapshot['bid_price_0'])
    ask = float(snapshot['ask_price_0'])
    return (bid + ask) / 2.0


def get_spread(snapshot: np.ndarray) -> float:
    """Calculate bid-ask spread from snapshot."""
    return float(snapshot['ask_price_0'] - snapshot['bid_price_0'])
