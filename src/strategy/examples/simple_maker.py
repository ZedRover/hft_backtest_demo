"""
Simple market maker strategy.

Places quotes at fixed spread around mid price.
Implements the example from quickstart.md.
"""

from typing import List
from ..base import BaseStrategy
from ..context import StrategyContext
from ..actions import SubmitOrder, OrderAction


class SimpleMaker(BaseStrategy):
    """
    Simple market-making strategy that places quotes at fixed spread.

    Strategy:
        - If no active orders: place buy and sell quotes
        - If have active orders: wait (don't add more)
        - Quotes placed at spread_bps basis points around mid

    Args:
        spread_bps: Target spread in basis points (e.g., 10 = 0.1%)
        quote_size: Size of each quote
        max_position: Maximum absolute position size
    """

    def __init__(
        self,
        spread_bps: int = 10,
        quote_size: float = 1.0,
        max_position: float = 10.0
    ):
        self.spread_bps = spread_bps
        self.quote_size = quote_size
        self.max_position = max_position

    def on_snapshot(self, context: StrategyContext) -> List[OrderAction]:
        """Generate order actions based on market state"""
        # Don't add more orders if we already have active orders
        if context.has_active_orders():
            return []

        # Don't trade if position limit reached
        if abs(context.position_quantity) >= self.max_position:
            return []

        # Calculate target prices
        mid = context.get_mid_price()
        spread = mid * self.spread_bps / 10000  # basis points to decimal

        buy_price = mid - spread / 2
        sell_price = mid + spread / 2

        # Place quotes on both sides
        actions: List[OrderAction] = [
            SubmitOrder(side=1, price=buy_price, quantity=self.quote_size),
            SubmitOrder(side=-1, price=sell_price, quantity=self.quote_size),
        ]

        return actions