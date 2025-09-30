"""
State management for backtest execution.

Tracks orders, positions, and fills during backtest.
"""

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
from ..data.types import ORDER_DTYPE, FILL_DTYPE, create_order_array, create_fill_array


@dataclass
class Position:
    """
    Current position state.

    Tracks quantity, average entry price, and P&L.
    """
    symbol: str
    quantity: float = 0.0  # Signed: positive=long, negative=short
    average_entry_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl

    def is_flat(self) -> bool:
        """Check if position is flat"""
        return abs(self.quantity) < 1e-9


class BacktestState:
    """
    Maintains state during backtest execution.

    Tracks:
    - Active orders
    - Position
    - Fills history
    - P&L series
    - Position series
    """

    def __init__(self, symbol: str, initial_capacity: int = 10000):
        """
        Initialize backtest state.

        Args:
            symbol: Trading symbol
            initial_capacity: Initial capacity for orders/fills arrays
        """
        self.symbol = symbol
        self.position = Position(symbol=symbol)

        # Order management
        self.orders: Dict[int, np.ndarray] = {}  # order_id -> order record
        self.next_order_id = 1

        # Fill tracking
        self.fills_list: List[np.ndarray] = []
        self.next_fill_id = 1

        # Time series
        self.timestamps: List[int] = []
        self.pnl_series: List[float] = []
        self.position_series: List[float] = []

        # Statistics
        self.total_orders = 0
        self.filled_orders = 0
        self.cancelled_orders = 0
        self.rejected_orders = 0

    def create_order(
        self,
        timestamp: int,
        side: int,
        price: float,
        quantity: float,
        queue_position: int,
        volume_ahead: float
    ) -> int:
        """
        Create a new order.

        Args:
            timestamp: Order submission time
            side: 1=buy, -1=sell
            price: Limit price
            quantity: Order quantity
            queue_position: Initial queue position
            volume_ahead: Initial volume ahead

        Returns:
            order_id
        """
        order_id = self.next_order_id
        self.next_order_id += 1

        # Create order record
        order = np.zeros(1, dtype=ORDER_DTYPE)[0]
        order['order_id'] = order_id
        order['timestamp'] = timestamp
        order['symbol'] = self.symbol
        order['side'] = side
        order['price'] = price
        order['quantity'] = quantity
        order['remaining_quantity'] = quantity
        order['filled_quantity'] = 0.0
        order['state'] = 0  # pending
        order['queue_position'] = queue_position
        order['volume_ahead'] = volume_ahead

        self.orders[order_id] = order
        self.total_orders += 1

        return order_id

    def get_active_orders(self) -> List[np.ndarray]:
        """
        Get all active orders (pending or partial).

        Returns:
            List of order records
        """
        active = []
        for order in self.orders.values():
            state = int(order['state'])
            if state in (0, 1):  # pending or partial
                active.append(order)
        return active

    def update_order_queue(
        self,
        order_id: int,
        new_queue_position: int,
        new_volume_ahead: float
    ) -> None:
        """
        Update order's queue position.

        Args:
            order_id: Order ID
            new_queue_position: New position in queue
            new_volume_ahead: New volume ahead
        """
        if order_id in self.orders:
            order = self.orders[order_id]
            order['queue_position'] = new_queue_position
            order['volume_ahead'] = new_volume_ahead

    def record_fill(
        self,
        order_id: int,
        timestamp: int,
        price: float,
        quantity: float,
        side: int,
        fee: float,
        is_maker: bool = True
    ) -> int:
        """
        Record a fill for an order.

        Args:
            order_id: Order ID
            timestamp: Fill timestamp
            price: Fill price
            quantity: Fill quantity
            side: Fill side
            fee: Transaction fee
            is_maker: True if maker fill

        Returns:
            fill_id
        """
        fill_id = self.next_fill_id
        self.next_fill_id += 1

        # Create fill record
        fill = np.zeros(1, dtype=FILL_DTYPE)[0]
        fill['fill_id'] = fill_id
        fill['order_id'] = order_id
        fill['timestamp'] = timestamp
        fill['price'] = price
        fill['quantity'] = quantity
        fill['side'] = side
        fill['fee'] = fee
        fill['is_maker'] = is_maker

        self.fills_list.append(fill)

        # Update order state
        if order_id in self.orders:
            order = self.orders[order_id]
            order['filled_quantity'] += quantity
            order['remaining_quantity'] -= quantity

            # Update state
            if order['remaining_quantity'] <= 0:
                order['state'] = 2  # filled
                self.filled_orders += 1
            else:
                order['state'] = 1  # partial

        return fill_id

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID

        Returns:
            True if cancelled, False if not found or already terminal
        """
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        state = int(order['state'])

        # Can only cancel pending or partial orders
        if state in (0, 1):  # pending or partial
            order['state'] = 3  # cancelled
            self.cancelled_orders += 1
            return True

        return False

    def reject_order(self, order_id: int) -> bool:
        """
        Reject an order.

        Args:
            order_id: Order ID

        Returns:
            True if rejected
        """
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        order['state'] = 4  # rejected
        self.rejected_orders += 1
        return True

    def update_position(
        self,
        new_quantity: float,
        new_avg_entry: float,
        new_realized_pnl: float
    ) -> None:
        """
        Update position after fill.

        Args:
            new_quantity: New position quantity
            new_avg_entry: New average entry price
            new_realized_pnl: New realized P&L
        """
        self.position.quantity = new_quantity
        self.position.average_entry_price = new_avg_entry
        self.position.realized_pnl = new_realized_pnl

    def update_unrealized_pnl(
        self,
        current_price: float,
        contract_multiplier: float
    ) -> None:
        """
        Update unrealized P&L based on current price.

        Args:
            current_price: Current market price
            contract_multiplier: Contract multiplier
        """
        if abs(self.position.quantity) > 1e-9:
            price_diff = current_price - self.position.average_entry_price
            self.position.unrealized_pnl = (
                price_diff * self.position.quantity * contract_multiplier
            )
        else:
            self.position.unrealized_pnl = 0.0

    def record_snapshot(self, timestamp: int) -> None:
        """
        Record state at a snapshot.

        Args:
            timestamp: Snapshot timestamp
        """
        self.timestamps.append(timestamp)
        self.pnl_series.append(self.position.total_pnl)
        self.position_series.append(self.position.quantity)

    def get_fills_array(self) -> np.ndarray:
        """
        Get all fills as NumPy array.

        Returns:
            Structured array of fills
        """
        if not self.fills_list:
            return np.array([], dtype=FILL_DTYPE)

        return np.array(self.fills_list, dtype=FILL_DTYPE)

    def get_orders_array(self) -> np.ndarray:
        """
        Get all orders as NumPy array.

        Returns:
            Structured array of orders
        """
        if not self.orders:
            return np.array([], dtype=ORDER_DTYPE)

        return np.array(list(self.orders.values()), dtype=ORDER_DTYPE)