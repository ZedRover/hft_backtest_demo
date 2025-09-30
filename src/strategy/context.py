"""
Strategy context.

Provides immutable context to strategies for decision-making.
"""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass(frozen=True)
class StrategyContext:
    """
    Immutable context passed to strategy at each snapshot.

    Contains all information strategy needs to make decisions:
    - Current market snapshot
    - Current position
    - Active orders
    """

    snapshot: np.ndarray  # Single snapshot record (structured array element)
    position_quantity: float  # Current position (signed)
    position_avg_entry: float  # Average entry price
    position_realized_pnl: float  # Realized P&L
    position_unrealized_pnl: float  # Unrealized P&L
    active_orders: List[np.ndarray]  # List of active order records
    timestamp: int  # Current timestamp (milliseconds)
    symbol: str  # Trading symbol

    def get_mid_price(self) -> float:
        """Calculate mid price from snapshot"""
        best_bid = self.snapshot['bid_price_0']
        best_ask = self.snapshot['ask_price_0']
        return (best_bid + best_ask) / 2.0

    def get_spread(self) -> float:
        """Calculate bid-ask spread"""
        best_bid = self.snapshot['bid_price_0']
        best_ask = self.snapshot['ask_price_0']
        return best_ask - best_bid

    def get_best_bid(self) -> float:
        """Get best bid price"""
        return self.snapshot['bid_price_0']

    def get_best_ask(self) -> float:
        """Get best ask price"""
        return self.snapshot['ask_price_0']

    def get_position_pnl(self) -> tuple[float, float]:
        """Get (realized, unrealized) P&L"""
        return self.position_realized_pnl, self.position_unrealized_pnl

    def get_total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)"""
        return self.position_realized_pnl + self.position_unrealized_pnl

    def has_active_orders(self) -> bool:
        """Check if strategy has any active orders"""
        return len(self.active_orders) > 0

    def count_active_orders_by_side(self, side: int) -> int:
        """
        Count active orders by side.

        Args:
            side: 1 for buy, -1 for sell

        Returns:
            Number of active orders on that side
        """
        return sum(1 for order in self.active_orders if order['side'] == side)